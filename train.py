import copy
import multiprocessing
import os
import sys
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pprint import pprint
from typing import Literal, Sized, TypeAlias, cast

import datasets
import datasets.config
import librosa
import matplotlib.pyplot as plt
import nnAudio.features
import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torchinfo
from gammatone.filters import centre_freqs, erb_filterbank, make_erb_filters
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info


class Config:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cpu_count = multiprocessing.cpu_count()

        self.epochs = 300  # 100

        self.lr = 0.002  # 1e-4
        self.weight_decay = 0.001  # 1e-5  # L2 regularisation
        self.momentum = 0.9
        self.nesterov_momentum = True  # ^ These are from the Piczak CNN paper

        self.batch_size = 200  # 24  # 64
        # self.criterion = nn.CrossEntropyLoss()
        # self.criterion_name = "Cross Entropy Loss"
        self.criterion = nn.NLLLoss()
        self.criterion_name = "Negative Log Loss"

        self.fold_count = 5
        self.test_fold_index = 3  # for 1-fold testing only

        self.input_shape: tuple[int, ...] = (-1, -1, -1)

        self.cache_dir = os.path.join(".cache")
        self.output_dir = os.path.join("output")

        self.display_graphics = True
        self.report_interval = 4

        self.model_path = sys.argv[1] if (len(sys.argv) == 2) else None
        # self.model_path = None  # use this for jupyter

        self.halt_accuracy_percentage: float | None = None

    def print(self) -> None:
        print("=" * 32)
        print("Config: ")
        for prop, value in vars(self).items():
            print(f"    {prop}: ", end="")
            pprint(value)
        print("=" * 32)
        print()


config = Config()

@contextmanager
def temporary_config():
    global config
    orig = copy.copy(config)
    yield config
    config = orig

class RawDataset(ABC):
    @abstractmethod
    def get_train_and_test_datasets(
        self, fold_count: int, test_fold_index: int
    ) -> tuple[Dataset[dict], Dataset[dict]]:
        pass

    @abstractmethod
    def get_class_count(self) -> int:
        pass


class DebugDataset(RawDataset):
    class CustomDataset(Dataset[dict]):
        def __init__(self, data: list[dict], repeat: int) -> None:
            super().__init__()
            self.data = data
            self.repeat = repeat
            assert repeat > 0

        def __len__(self) -> int:
            return len(self.data) * self.repeat

        def __getitem__(self, index) -> dict:
            return self.data[index % len(self.data)]

    def __init__(self, class_count: int, size: int = 224) -> None:
        self.class_count = class_count
        self.data: list[dict] = [
            {
                "audio": torch.tensor(
                    np.pad(
                        np.array([[1]]),
                        ((i, size - 1 - i), (i, size - 1 - i)),
                        mode="constant",
                        constant_values=0,
                    ),
                    dtype=torch.float,
                ).unsqueeze(0),
                "target": i,
            }
            for i in range(class_count)
        ]

        test_sample: torch.Tensor = self.data[min(1, class_count - 1)]["audio"]
        print("Debug Dataset: shape =", test_sample.size())
        print("e.g.:", test_sample)
        assert test_sample.size() == (1, size, size)

    def get_class_count(self) -> int:
        return self.class_count

    def get_train_and_test_datasets(
        self, fold_count: int, test_fold_index: int
    ) -> tuple[Dataset[dict], Dataset[dict]]:
        dataset = self.CustomDataset(self.data, 16)
        return dataset, dataset


class EscDataset(RawDataset):
    """ESC-50/ESC-10 Dataset"""

    Format: TypeAlias = Literal["full", "seg-long", "seg-short"]

    esc10_label_map = {
        0: (0, "dog"),
        10: (1, "rain"),
        11: (2, "waves"),
        20: (3, "cry"),
        38: (4, "clock"),
        21: (5, "sneeze"),
        40: (6, "helicopter"),
        41: (7, "chainsaw"),
        1: (8, "rooster"),
        12: (9, "fire"),
    }

    esc10_cache_path = os.path.join(config.cache_dir, "esc10", "{}", "{}", "fold-{}")
    esc50_cache_path = os.path.join(config.cache_dir, "esc50", "{}", "{}", "fold-{}")
    hugging_face_cache_path = os.path.join(config.cache_dir, "datasets")

    def __init__(self, *, esc10: bool) -> None:
        self.esc10 = esc10
        if self.are_test_and_train_format_different():
            self.test_folds = self.load_dataset(
                esc10, fold_count=5, is_test_dataset=True
            )
            self.train_folds = self.load_dataset(
                esc10, fold_count=5, is_test_dataset=False
            )
        else:
            self.train_folds = self.test_folds = self.load_dataset(esc10, fold_count=5)

    def load_dataset(
        self, esc10: bool, fold_count: int, is_test_dataset: bool | None = None
    ) -> list[datasets.Dataset]:
        esc_cache_path = self.esc10_cache_path if esc10 else self.esc50_cache_path
        prepared_dataset: datasets.Dataset | None = None

        # is_included = lambda row: row["esc10"] or not esc10

        dataset_folds: list[datasets.Dataset] = []

        for fold_index in range(fold_count):
            is_current_fold = lambda row: row["fold"] - 1 == fold_index

            save_folder_name = self.get_format() + ("-test" if is_test_dataset else "")
            fold_save_path = esc_cache_path.format(
                self.get_name(), save_folder_name, fold_index
            )
            if os.path.exists(fold_save_path):
                existing = datasets.Dataset.load_from_disk(fold_save_path)
                print(
                    f"ESC Dataset: Found cache for fold-{fold_index} at '{fold_save_path}': "
                    + f"sample count = {len(existing)}"
                )
                dataset_folds.append(existing)
                continue

            if prepared_dataset is None:
                raw_dataset = self.fetch_raw_dataset()
                if esc10:
                    # Remove non-ESC-10 entries first
                    raw_dataset = raw_dataset.filter(
                        lambda flag: flag,
                        batch_size=64,
                        num_proc=config.cpu_count,
                        input_columns="esc10",
                    )
                # Lazily prepare dataset
                prepared_dataset = self.prepare_dataset(raw_dataset, is_test_dataset)

            fold_dataset = prepared_dataset.filter(
                # Here be 24 hours of pain because of a `not`
                lambda row: is_current_fold(row),
                batch_size=64,
                num_proc=config.cpu_count,
            )

            fold_dataset.save_to_disk(fold_save_path)
            print(f"ESC Dataset: Saved fold-{fold_index} to '{fold_save_path}'.")

            dataset_folds.append(fold_dataset)

        assert len(dataset_folds) == fold_count

        return dataset_folds

    @classmethod
    def fetch_raw_dataset(cls) -> datasets.Dataset:
        # import dataset from https://huggingface.co/datasets/ashraq/esc50
        return cast(
            datasets.Dataset,
            datasets.load_dataset(
                "ashraq/esc50", split="train", cache_dir=cls.hugging_face_cache_path
            ),
        )

    @abstractmethod
    def get_format(self) -> Format:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def prepare_dataset(
        self, dataset: datasets.Dataset, is_test_dataset: bool | None
    ) -> datasets.Dataset:
        pass

    @abstractmethod
    def are_test_and_train_format_different(self) -> bool:
        return False

    def clean_and_resample(
        self, dataset: datasets.Dataset, sample_rate: int
    ) -> datasets.Dataset:
        dataset = dataset.remove_columns(
            # ["filename", "category", "src_file", "take"]
            ["category", "src_file", "take"]
        )

        # Resample audio
        dataset = dataset.cast_column(
            "audio", datasets.Audio(sampling_rate=sample_rate)
        )

        print(
            f"Preprocess: audio resampled to {dataset[0]['audio']['sampling_rate']}Hz "
            + f"({len(dataset[0]['audio']['array'])} frames)"
        )

        return dataset

    def get_train_and_test_datasets(
        self, fold_count: int, test_fold_index: int
    ) -> tuple[Dataset[dict], Dataset[dict]]:
        assert fold_count == 5  # Mandatory requirement...

        train_dataset = datasets.concatenate_datasets(
            [d for i, d in enumerate(self.train_folds) if i != test_fold_index]
        )

        test_dataset = self.test_folds[test_fold_index]

        # self.dump_splits_to_file(
        #     os.path.join(config.output_dir, "split_dump.txt"),
        #     train_dataset,
        #     test_dataset,
        # )
        # exit()

        return self.convert_train_and_test_datasets(train_dataset, test_dataset)

    def convert_train_and_test_datasets(
        self, train_dataset: datasets.Dataset, test_dataset: datasets.Dataset
    ) -> tuple[Dataset[dict], Dataset[dict]]:
        """Convert to PyTorch datasets"""
        return (
            # Still have to map to Mel Spectrograms
            cast(
                Dataset[dict],
                train_dataset.with_format("torch"),
            ),
            cast(
                Dataset[dict],
                test_dataset.with_format("torch"),
            ),
        )

    def get_class_count(self) -> int:
        return 10 if self.esc10 else 50

    def get_sample_rate(self) -> int:
        return self.train_folds[0][0]["audio"]["sampling_rate"]

    def convert_target(self, target: int) -> int:
        return self.esc10_label_map.get(target, (-1, None))[0] if self.esc10 else target

    @staticmethod
    def dump_splits_to_file(
        filename: str, train: datasets.Dataset, test: datasets.Dataset
    ):
        def dump_line(row: dict) -> str:
            return (
                str(row["filename"])
                + " -> "
                + str(row["target"])
                + ": "
                + str(np.shape(row["audio"]))
                + "\n"
            )

        with open(filename, "w") as file:
            file.write("Train:\n\n")
            file.writelines((dump_line(cast(dict, row)) for row in train))

            file.write("\n\nTest:\n\n")
            file.writelines((dump_line(cast(dict, row)) for row in test))
            file.write("\n")

        print(f"Dumped splits to '{filename}'")


class EscFullMelDataset(EscDataset):
    sample_rate = 15_670  # 16_000
    mel_n_fft = 1024
    mel_hop_length = 350  # 256
    mel_count = 224  # 128

    def __init__(self, *, esc10: bool) -> None:
        assert config.input_shape == (1, 224, 224)
        super().__init__(esc10=esc10)

    def get_name(self) -> str:
        return "mel"

    def get_format(self) -> EscDataset.Format:
        return "full"

    def prepare_dataset(
        self, dataset: datasets.Dataset, is_test_dataset: bool | None = None
    ) -> datasets.Dataset:
        """Pre-processing pipeline"""

        # pipeline_mapper = lambda audio_batch: {
        #     "audio": pipeline(
        #         torch.tensor(
        #             np.array([audio["array"] for audio in audio_batch]),
        #             dtype=torch.float,
        #             device=config.device,
        #         )
        #     )
        # }

        assert is_test_dataset is None

        dataset = self.clean_and_resample(dataset, self.sample_rate)

        # CPU spectrogram is actually faster
        # (may be due to not having to copy tensors between GPU and CPU)
        data_transformer = lambda audio, target: {
            "audio": np.expand_dims(
                # librosa.power_to_db(
                (
                    librosa.feature.melspectrogram(
                        y=audio["array"],
                        sr=self.sample_rate,
                        n_fft=self.mel_n_fft,
                        hop_length=self.mel_hop_length,
                        n_mels=self.mel_count,
                    )
                ),
                0,  # One channel -> unsqueeze array of size (mel, time) to (1, mel, time)
            ),
            "target": self.convert_target(target),
        }

        # Pass audio waveforms though our processing pipline
        dataset = dataset.map(
            # pipeline_mapper,
            data_transformer,
            batch_size=128,
            # batched=True,
            input_columns=["audio", "target"],
            # remove_columns=["filename", "category", "src_file", "take"],
            num_proc=config.cpu_count,
        )

        return dataset


class EscSegmentDataset(EscDataset):
    sample_rate = 22_050

    # ESC-10:
    esc10_limits = {
        1: ((-4, 4), (0.95, 1.1)),
        3: ((0, 0), (0.9, 1.2)),
        4: ((-3, 6), (0.8, 1.3)),
        6: ((-4, 4), (0.9, 1.2)),
        7: ((0, 0), (0.9, 1.2)),
        8: ((-4, 2), (0.9, 1.2)),
        9: ((-3, 2), (0.95, 1.1)),
    }

    def __init__(self, *, esc10: bool, long: bool, augment_count: int, use_test_format: bool) -> None:
        self.long = long
        self.augment_count = augment_count
        self.frames_per_segment = 101 if self.long else 41
        self.use_test_format = use_test_format
        self.test_dataset: datasets.Dataset | None = None  # the cached test dataset
        super().__init__(esc10=esc10)

    def load_dataset(
        self, esc10: bool, fold_count: int, is_test_dataset: bool | None = None
    ) -> list[datasets.Dataset]:
        if self.use_test_format:
            is_test_dataset = True
        return super().load_dataset(esc10, fold_count, is_test_dataset)

    def prepare_dataset(
        self, dataset: datasets.Dataset, is_test_dataset: bool | None
    ) -> datasets.Dataset:
        if self.augment_count > 0:
            return self.prepare_dataset_with_augment(dataset, is_test_dataset)
        return self.prepare_dataset_no_augment(dataset, is_test_dataset)

    def prepare_dataset_with_augment(
        self, dataset: datasets.Dataset, is_test_dataset: bool | None
    ) -> datasets.Dataset:
        assert is_test_dataset is not None

        dataset = self.clean_and_resample(dataset, self.sample_rate)

        print(
            f"Preprocessing: cleaned dataset has {len(dataset)} samples (test dataset = {is_test_dataset})."
        )

        if is_test_dataset:

            def test_data_transformer(audio: dict, target: int, filename: str) -> dict:
                segments, ranges = self.extract_segments(filename, audio["array"])
                return {
                    "audio": segments,
                    "ranges": ranges,
                    "target": self.convert_target(target),
                }

            return dataset.map(
                test_data_transformer,
                input_columns=["audio", "target", "filename"],
                num_proc=config.cpu_count,
            )
        else:

            def train_data_transformer(rows: dict[str, list]) -> dict[str, list]:
                assert len(rows["audio"]) == 1
                audio = rows["audio"][0]["array"]
                target = self.convert_target(rows["target"][0])
                segments = self.extract_segments(rows["filename"][0], audio)[0]
                for _ in range(self.augment_count):
                    augmented = self.augment(audio, target)
                    segments.extend(
                        self.extract_segments(rows["filename"][0], augmented)[0]
                    )
                out_count = len(segments)
                remaining = {
                    key: [value[0]] * out_count
                    for key, value in rows.items()
                    if key not in {"audio", "target"}
                }
                return {
                    "audio": segments,
                    "target": [target] * out_count,
                    **remaining,
                }

            return dataset.map(
                train_data_transformer,
                batch_size=1,  # Using batched to split dataset only
                batched=True,
                num_proc=config.cpu_count,
            )

    def prepare_dataset_no_augment(
        self, dataset: datasets.Dataset, is_test_dataset: bool | None
    ) -> datasets.Dataset:
        assert is_test_dataset is not None

        if self.test_dataset is None:
            dataset = self.clean_and_resample(dataset, self.sample_rate)
            print(f"Preprocessing: cleaned dataset has {len(dataset)} samples.")

            def test_data_transformer(audio: dict, target: int, filename: str) -> dict:
                segments, ranges = self.extract_segments(filename, audio["array"])
                return {
                    "audio": segments,
                    "ranges": ranges,
                    "target": self.convert_target(target),
                }

            self.test_dataset = dataset.map(
                test_data_transformer,
                input_columns=["audio", "target", "filename"],
                num_proc=config.cpu_count,
            )

        if is_test_dataset:
            dataset = self.test_dataset
        else:

            def train_data_transformer(rows: dict[str, list]) -> dict[str, list]:
                assert len(rows["audio"]) == 1
                segments = rows["audio"][0]
                out_count = len(segments)
                remaining = {
                    key: [value[0]] * out_count
                    for key, value in rows.items()
                    if key not in {"audio", "ranges"}
                }
                return {
                    "audio": segments,
                    **remaining,
                }

            dataset = self.test_dataset.map(
                train_data_transformer,
                batch_size=1,  # Using batched to split dataset only
                batched=True,
                num_proc=config.cpu_count,
                remove_columns=["ranges"],
            )

        print(f"Preprocessing: prepared dataset now has {len(dataset)} samples.")

        return dataset
    
    @abstractmethod
    def extract_segments(
        self, name: str, clip: np.ndarray
    ) -> tuple[list[np.ndarray], list[tuple]]:
        pass

    @abstractmethod
    def augment(self, audio: np.ndarray, target: int) -> np.ndarray:
        pass


class EscLogMelSegmentDataset(EscSegmentDataset):
    def __init__(self, *, esc10: bool, long: bool, augment_count=0, use_test_format=False) -> None:
        super().__init__(esc10=esc10, long=long, augment_count=augment_count, use_test_format=use_test_format)

    def get_name(self) -> str:
        aug_postfix = ("-aug-" + str(self.augment_count)) if self.augment_count else ""
        return "log-mel" + aug_postfix

    def get_format(self) -> EscDataset.Format:
        return "seg-long" if self.long else "seg-short"

    def extract_segments(
        self, name: str, clip: np.ndarray
    ) -> tuple[list[np.ndarray], list[tuple]]:
        # Based on https://github.com/karolpiczak/paper-2015-esc-convnet/blob/master/Code/_Datasets/Setup.ipynb

        # Due to an off-by-one bug which has not been caught earlier,
        # actually both variants (long and short) use the same
        # overlap setting (half of window size) - whereas different settings
        # were mentioned in the paper.
        #
        # The code below has been already cleaned up to reflect those changes.
        #
        # Apart from that, for reproducibility purposes it is required that
        # librosa v0.3.1 is used, as further versions drastically change
        # the delta computations.
        #
        #   -- Note from the original source code.

        # But let's just use the latest `librosa` and hope it still works.

        FRAMES_PER_SEGMENT = (
            self.frames_per_segment - 1
        )  # 41 frames ~= 950 ms segment length @ 22050 Hz
        WINDOW_SIZE = 512 * FRAMES_PER_SEGMENT  # 23 ms per frame @ 22050 Hz
        STEP_SIZE = 512 * int(FRAMES_PER_SEGMENT * (0.1 if self.long else 0.5))
        BANDS = 60

        s = 0
        segments = []
        ranges = []

        normalization_factor = 1 / np.max(np.abs(clip))
        clip = clip * normalization_factor

        while (
            len(signal := clip[s * STEP_SIZE : s * STEP_SIZE + WINDOW_SIZE])
            == WINDOW_SIZE
        ):
            melspec = librosa.feature.melspectrogram(
                y=signal, sr=self.sample_rate, n_fft=1024, hop_length=512, n_mels=BANDS
            )

            logspec = librosa.power_to_db(melspec)

            # # [[1,2],[3,4]] -> [[1,2,3,4]] (maybe)
            # logspec = logspec.T.flatten()[:, np.newaxis].T

            # ^ I replaced the above code with the code below... Hopefully won't break anything.

            delta = librosa.feature.delta(logspec)

            if np.mean(logspec) > -70.0:  # drop silent frames
                segments.append([logspec, delta])
                ranges.append((s * STEP_SIZE, s * STEP_SIZE + WINDOW_SIZE))

            s += 1

        if len(segments) <= 0:
            # For debugging
            print("\n\n" + name + "\n\n")
        assert len(segments) > 0

        return segments, ranges

    def augment(self, audio: np.ndarray, target: int) -> np.ndarray:
        limits = ((0, 0), (1.0, 1.0))  # pitch shift in half-steps, time stretch

        if self.esc10:
            limits = self.esc10_limits.get(target, limits)

        pitch_shift = np.random.randint(limits[0][1], limits[0][1] + 1)
        time_stretch = np.random.random() * (limits[1][1] - limits[1][0]) + limits[1][0]
        time_shift = np.random.randint(self.sample_rate)

        shifted = librosa.effects.pitch_shift(
            audio, sr=self.sample_rate, n_steps=pitch_shift
        )
        stretched = librosa.effects.time_stretch(shifted, rate=time_stretch)

        return np.hstack(
            (np.zeros((time_shift)), stretched)
        )  # pad a random amount of zeroes in front

    def are_test_and_train_format_different(self) -> bool:
        return not self.use_test_format

class EscAugmentedLogMelSegmentDataset(EscLogMelSegmentDataset):
    sample_rate = 22_050

    # ESC-10:
    esc10_limits = {
        1: ((-4, 4), (0.95, 1.1)),
        3: ((0, 0), (0.9, 1.2)),
        4: ((-3, 6), (0.8, 1.3)),
        6: ((-4, 4), (0.9, 1.2)),
        7: ((0, 0), (0.9, 1.2)),
        8: ((-4, 2), (0.9, 1.2)),
        9: ((-3, 2), (0.95, 1.1)),
    }

    class OurIterableTrainingDataset(IterableDataset[dict]):
        def __init__(self, outer: 'EscAugmentedLogMelSegmentDataset', dataset: Dataset[dict]):
            super(EscAugmentedLogMelSegmentDataset.OurIterableTrainingDataset).__init__()
            self.outer = outer
            self.dataset = dataset

        def __iter__(self):
            worker_info = get_worker_info()
            assert worker_info is None # single-process data loading
            return self.generate_data()

        def generate_data(self):
            for entry in self.dataset:
                name = entry["filename"]
                waveform: torch.Tensor = entry["audio"]["array"]
                augmented = self.outer.augment(waveform.numpy(), entry["target"])
                segments, ranges = self.outer.extract_segments(name, augmented)
                for segment in segments:
                    segment_tensor = torch.tensor(segment, device=waveform.device)
                    yield { **entry, "audio": segment_tensor }

    def __init__(self, *, esc10: bool, long: bool) -> None:
        super().__init__(esc10=esc10, long=long, augment_count=0, use_test_format=False)

    def get_name(self) -> str:
        return "log-mel-aug"

    def get_format(self) -> EscDataset.Format:
        return "seg-long" if self.long else "seg-short"

    def prepare_dataset(
        self, dataset: datasets.Dataset, is_test_dataset: bool | None
    ) -> datasets.Dataset:
        assert is_test_dataset is not None

        dataset = self.clean_and_resample(dataset, self.sample_rate)
        print(f"Preprocessing: cleaned dataset has {len(dataset)} samples.")

        if is_test_dataset:

            def test_data_transformer(audio: dict, target: int, filename: str) -> dict:
                segments, ranges = self.extract_segments(filename, audio["array"])
                return {
                    "audio": segments,
                    "ranges": ranges,
                    "target": self.convert_target(target),
                }

            dataset = dataset.map(
                test_data_transformer,
                input_columns=["audio", "target", "filename"],
                num_proc=config.cpu_count,
            )
        else:

            def train_data_transformer(target: int) -> dict:
                # Note how we keep the raw audio waveform untouched
                return {
                    "target": self.convert_target(target),
                }

            dataset = dataset.map(
                train_data_transformer,
                input_columns=["target"],
                num_proc=config.cpu_count,
            )

        print(f"Preprocessing: prepared dataset now has {len(dataset)} samples.")
        return dataset

    def convert_train_and_test_datasets(
        self, train_dataset: datasets.Dataset, test_dataset: datasets.Dataset
    ) -> tuple[Dataset[dict], Dataset[dict]]:
        train, test = super().convert_train_and_test_datasets(train_dataset, test_dataset)
        return self.OurIterableTrainingDataset(self, train), test

    def are_test_and_train_format_different(self) -> bool:
        return True

class EscFusedTeoGtscDataset(EscSegmentDataset):
    """Inheritance is messed up -- TODO: extract superclass later..."""

    def __init__(self, *, esc10: bool) -> None:
        super().__init__(esc10=esc10, long=False, augment_count=0, use_test_format=False)

    def get_name(self) -> str:
        return "gtsc"

    def get_format(self) -> EscDataset.Format:
        return "seg-short"

    sample_rate = EscSegmentDataset.sample_rate
    centre_f = centre_freqs(sample_rate, 60, 20)
    erb_filters = make_erb_filters(sample_rate, centre_f)
    fcoefs = np.flipud(erb_filters)

    # Saw this method here: https://github.com/pratyush-prateek/environmental_sound_classification_1DCNN/blob/e9e43fc868d8a137b2f157ac84f3dc80ae3c054a/model_config.py#L147
    # which holds the source code for the paper https://doi.org/10.1016/j.eswa.2019.06.040
    # (End-to-end environmental sound classification using a 1D convolutional neural network)
    # gram = nnAudio.features.Gammatonegram(
    #     sr=ESCLogMelSegmentDataset.sample_rate,
    #     n_bins=60,
    #     trainable_STFT=False,
    #     verbose=False,
    # )

    @classmethod
    def apply_gammatone_filterbank(cls, y, sr: int, num_bands):
        # return cls.gram(torch.tensor(y).float()).squeeze().numpy()
        return erb_filterbank(y, cls.fcoefs)
        # TODO: Not sure about this part -- original split into hamming window
        # and dropped continuously silent windows first

    # Half Wave Rectifier (HWR)
    @staticmethod
    def half_wave_rectifier(y):
        return np.maximum(y, 0)

    # Teager Energy Operator (TEO)
    @staticmethod
    def teager_energy_operator(y: np.ndarray):
        energy = y[1:-1] ** 2 - y[:-2] * y[2:]
        # When the central sample is very small (or close to zero)
        # and the surrounding samples are NON-zero,
        # e.g. [1, 1, 0, 1, 1] -> [1, -1, 1]
        # So we replace the negative values with zero.
        energy[energy < 0] = 0
        return energy

    # Short-term Averaging
    @staticmethod
    def short_term_averaging(signal: np.ndarray, window_size=1024, hop_size=512):
        window_count = (len(signal) - window_size) // hop_size + 1
        result = np.zeros(window_count)

        for i in range(window_count):
            start_index = i * hop_size
            end_index = start_index + window_size
            window = signal[start_index:end_index]
            result[i] = np.mean(window)

        return result

    def extract_segments(
        self, name: str, clip: np.ndarray
    ) -> tuple[list[np.ndarray], list[tuple]]:
        FRAMES_PER_SEGMENT = (
            self.frames_per_segment  # - 1 # keep off-by-one eerror for comparison ACTUALLY NVM :(
        )  # 41 frames ~= ??? ms segment length @ 22050 Hz (TODO)
        WINDOW_SIZE = FRAMES_PER_SEGMENT  # ?? ms per frame @ 22050 Hz
        STEP_SIZE = FRAMES_PER_SEGMENT // 2
        BANDS = 60

        # For score-level fusion
        alpha = 0.5

        s = 0
        segments = []
        ranges = []

        normalization_factor = 1 / np.max(np.abs(clip))
        clip = clip * normalization_factor

        # y_filtered = librosa.feature.melspectrogram(
        #     y=clip, sr=self.sample_rate, n_fft=1024, hop_length=512, n_mels=BANDS
        # )

        # y_log = librosa.power_to_db(y_filtered)

        # Apply Gammatone filterbank
        y_filtered = self.apply_gammatone_filterbank(clip, self.sample_rate, BANDS)
        # print(y_filtered.shape, np.min(y_filtered), np.max(y_filtered))

        # Get Gammatone-Gram by converting to power
        # y_spec = np.sqrt(np.apply_along_axis(self.short_term_averaging, 1, np.square(y_filtered)))
        y_spec = np.apply_along_axis(
            self.short_term_averaging, 1, np.square(y_filtered)
        )

        # Apply Half Wave Rectifier
        y_hwr = self.half_wave_rectifier(y_filtered)
        # print(y_hwr.shape, np.min(y_hwr), np.max(y_hwr))

        # Apply Teager Energy Operator
        y_teo = np.apply_along_axis(self.teager_energy_operator, 1, y_hwr)
        # print(y_teo.shape, np.min(y_teo), np.max(y_teo))

        # Apply Short-term Averaging
        y_sta = np.apply_along_axis(self.short_term_averaging, 1, y_teo)  # y_teo)
        # print(y_sta.shape, np.min(y_sta), np.max(y_sta))

        # Apply Logarithmic Compression
        # NOTE: meant to use log(1 + x), but pure log matches the diagram more so yea idk
        # y_log = np.apply_along_axis(np.log1p, 1, y_sta)
        # y_log = librosa.power_to_db(y_sta)
        y_log = np.log1p(np.sqrt(y_sta)) # nvm this should work... no??
        # print(y_log.shape, np.min(y_log), np.max(y_log))

        def normalise(y: np.ndarray):
            normalization_factor: float = 1 / np.max(np.abs(y))
            # print("Normalisation factor:", normalization_factor)
            return y * normalization_factor  # - 0.5

        fused = normalise(y_spec) * alpha + normalise(y_log) * (1 - alpha)

        # librosa.feature.melspectrogram

        # print(y_filtered.shape, y_log.shape)
        # exit()

        while (segment := fused[:, s * STEP_SIZE : s * STEP_SIZE + WINDOW_SIZE]).shape[
            1
        ] == WINDOW_SIZE:
            delta = librosa.feature.delta(segment)

            if np.mean(segment) > 1e-5:
                segments.append([segment, delta])
                ranges.append((s * STEP_SIZE, s * STEP_SIZE + WINDOW_SIZE))
            # else drop silent frames;
            # TODO: idk about the threshold -- i just guessed a random number :/

            s += 1

        if len(segments) <= 0:
            # For debugging
            print("\n\n" + name + "\n\n")
        assert len(segments) > 0

        return segments, ranges
    
    def augment(self, audio: np.ndarray, target: int) -> np.ndarray:
        raise NotImplementedError("Unsupported for now")
    
    def are_test_and_train_format_different(self) -> bool:
        return True


class EscPureTeoGtscDataset(EscSegmentDataset):
    """Inheritance is messed up -- TODO: extract superclass later..."""

    def __init__(self, *, esc10: bool, long: bool, use_test_format = False) -> None:
        super().__init__(esc10=esc10, long=long, augment_count=0, use_test_format=use_test_format)

    def get_name(self) -> str:
        return "teo-gtsc" # "ggram"

    sample_rate = EscLogMelSegmentDataset.sample_rate
    centre_f = centre_freqs(sample_rate, 60, 20)
    erb_filters = make_erb_filters(sample_rate, centre_f)
    fcoefs = np.flipud(erb_filters)

    # Saw this method here: https://github.com/pratyush-prateek/environmental_sound_classification_1DCNN/blob/e9e43fc868d8a137b2f157ac84f3dc80ae3c054a/model_config.py#L147
    # which holds the source code for the paper https://doi.org/10.1016/j.eswa.2019.06.040
    # (End-to-end environmental sound classification using a 1D convolutional neural network)
    # gram = nnAudio.features.Gammatonegram(
    #     sr=ESCLogMelSegmentDataset.sample_rate,
    #     n_bins=60,
    #     trainable_STFT=False,
    #     verbose=False,
    # )

    @classmethod
    def apply_gammatone_filterbank(cls, y):
        # return cls.gram(torch.tensor(y).float()).squeeze().numpy()
        return erb_filterbank(y, cls.fcoefs)
        # TODO: Not sure about this part -- original split into hamming window
        # and dropped continuously silent windows first

    # Half Wave Rectifier (HWR)
    @staticmethod
    def half_wave_rectifier(y):
        return np.maximum(y, 0)

    # 
    @staticmethod
    def teager_energy_operator(y: np.ndarray):
        energy = y[1:-1] ** 2 - y[:-2] * y[2:]
        # When the central sample is very small (or close to zero)
        # and the surrounding samples are NON-zero,
        # e.g. [1, 1, 0, 1, 1] -> [1, -1, 1]
        # So we replace the negative values with zero.
        energy[energy < 0] = 0
        return energy

    # Short-term Averaging
    @staticmethod
    def short_term_averaging(window_size: int, hop_size: int):
        def sta(signal: np.ndarray) -> np.ndarray:
            window_count = (len(signal) - window_size) // hop_size + 1
            result = np.zeros(window_count)

            for i in range(window_count):
                start_index = i * hop_size
                end_index = start_index + window_size
                window = signal[start_index:end_index]
                result[i] = np.mean(window)

            return result
        return sta

    def extract_segments(
        self, name: str, clip: np.ndarray
    ) -> tuple[list[np.ndarray], list[tuple]]:
        FRAMES_PER_SEGMENT = (
            self.frames_per_segment  # - 1
        )  # 41 frames ~= ??? ms segment length @ 22050 Hz (TODO)
        WINDOW_SIZE = FRAMES_PER_SEGMENT  # ?? ms per frame @ 22050 Hz
        STEP_SIZE = int(FRAMES_PER_SEGMENT * (0.1 if self.long else 0.5))
        MEL_BANDS = 60
        N_FFT = 1024
        HOP_LEN = N_FFT // 2

        # For non-concatenating feature-level fusion (not in paper)
        alpha = 0.5

        s = 0
        segments = []
        ranges = []

        normalization_factor = 1 / np.max(np.abs(clip))
        clip = clip * normalization_factor

        # Apply Gammatone filterbank
        y_filtered = self.apply_gammatone_filterbank(clip)
        # print(y_filtered.shape, np.min(y_filtered), np.max(y_filtered))

        # Apply Half Wave Rectifier
        y_hwr = self.half_wave_rectifier(y_filtered)
        # print(y_hwr.shape, np.min(y_hwr), np.max(y_hwr))

        # Apply Teager Energy Operator
        y_teo = np.apply_along_axis(self.teager_energy_operator, 1, y_hwr)
        # print(y_teo.shape, np.min(y_teo), np.max(y_teo))

        # Apply Short-term Averaging
        y_sta = np.apply_along_axis(self.short_term_averaging(N_FFT, HOP_LEN), 1, y_teo)  # y_teo)
        # print(y_sta.shape, np.min(y_sta), np.max(y_sta))

        # Apply Logarithmic Compression
        y_log = librosa.power_to_db(y_sta, ref=np.max, top_db=None) # np.log1p(y_sta * ((np.e - 1) / np.max(y_sta)))
        y_log /= np.max(np.abs(y_log))
        y_log += 1
        # print(y_log.shape, np.min(y_log), np.max(y_log))

        teo_gtsc = y_log

        fused = teo_gtsc


        # mel = librosa.feature.melspectrogram(
        #     y=clip, sr=self.sample_rate, n_fft=1024, hop_length=512, n_mels=MEL_BANDS
        # )

        # mel = librosa.power_to_db(mel, ref=np.max, top_db=None)
        # mel = mel[:, 1:-1]
        # mel /= np.max(np.abs(mel))
        # mel += 1

        # fused = alpha * teo_gtsc + (1 - alpha) * mel

        # y_spec = self.gammatone(torch.tensor(clip).float()).squeeze().numpy()
        # y_spec = librosa.power_to_db(y_spec, ref=np.max, top_db=None)
        # y_spec /= np.max(np.abs(y_spec))
        # y_spec += 1

        # fused = y_spec

        # hamming_window = np.hamming(WINDOW_SIZE)
        while (segment := fused[:, s * STEP_SIZE : s * STEP_SIZE + WINDOW_SIZE]).shape[
            1
        ] == WINDOW_SIZE:
            if np.mean(segment) > 0.025:
                # segment = segment * hamming_window
                delta = librosa.feature.delta(segment)
                segments.append([segment, delta])
                # mel_segment = mel[:, s * STEP_SIZE : s * STEP_SIZE + WINDOW_SIZE]
                # segments.append([segment, mel_segment])
                ranges.append((s * STEP_SIZE, s * STEP_SIZE + WINDOW_SIZE))
            # else drop silent frames;
            # TODO: idk about the threshold -- i just guessed a random number :/

            s += 1

        if len(segments) <= 0:
            # For debugging
            print("\n\n" + name + " " + str(np.mean(segment)) + "\n\n")
            np.savetxt(os.path.join(config.output_dir, "segment_dump.txt"), fused, "%.2e")
        assert len(segments) > 0

        return segments, ranges
    
    def augment(self, audio: np.ndarray, target: int) -> np.ndarray:
        raise NotImplementedError("Unsupported for now")
    
    def are_test_and_train_format_different(self) -> bool:
        return True


# class OurPipeline(nn.Module):
#     """Pre-processing pipeline on GPU. Unused for now."""

#     def __init__(self, orig_freq: int) -> None:
#         super().__init__()
#         self.orig_freq = orig_freq
#         self.resampler = T.Resample(orig_freq=orig_freq, new_freq=config.sample_rate)
#         self.spectrogram = T.MelSpectrogram(
#             config.sample_rate,
#             config.mel_n_fft,
#             config.mel_hop_length,
#             config.mel_count,
#         )

#     def forward(self, waveform: torch.Tensor) -> torch.Tensor:
#         assert waveform.dim() == 2 or waveform.dim() == 3
#         if waveform.dim() == 3:
#             # Mono conversion -- for other datasets; ESC is already mono
#             # print("Multi-channel audio found, coverting to mono...")
#             waveform = torch.mean(waveform, dim=1)

#         if self.orig_freq == config.sample_rate:
#             waveform = self.resampler(waveform)
#             # resampled.size() == (batch size, 80_000)

#         spectrogram: torch.Tensor = self.spectrogram(waveform)
#         # spectrogram.size() == (batch size, n_mel, spectrogram time length)

#         # for 1-channel, now has size (batch size, 1, ...)
#         return spectrogram.unsqueeze(1)


def init_weights(net: nn.Module) -> None:
    def init_pass(module: nn.Module) -> None:
        # Taken form torchvision.models.VGG
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0, 0.01)
            nn.init.constant_(module.bias, 0)

    net.apply(init_pass)
    print("Weight initialisation complete.")


# class DefaultOptimizerMixin(ABC):
#     def get_optimizer(self) -> torch.optim.Optimizer:
#         return optim.SGD(
#             self.parameters(),
#             lr=config.lr,
#             weight_decay=config.weight_decay,
#             momentum=config.momentum,
#         )

#     @abstractmethod
#     def parameters(self) -> Iterator[nn.Parameter]:
#         pass


class DebugNet(nn.Module):
    def __init__(self, class_count: int):
        super(DebugNet, self).__init__()
        self.main = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(np.product(config.input_shape).item(), 64),
            nn.Tanh(),
            nn.Linear(64, class_count),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


# based off VGG-16: https://www.geeksforgeeks.org/vgg-16-cnn-model/
class VGG16(nn.Module):
    """Our network (name yet to be decided)"""

    def __init__(self, class_count: int, input_channels=1):
        super(VGG16, self).__init__()
        assert config.input_shape == (input_channels, 224, 224)
        self.main = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=64,
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),
            nn.Flatten(start_dim=1),
            nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Linear(4096, class_count),
                nn.LogSoftmax(dim=1),
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE: x is always of size (batch size, ...)
        return self.main(x)


class TEOGammmatoneCNN(nn.Module):
    """CNN Proposed in "Novel TEO-based Gammatone Features for Environmental Sound Classification"

    D. M. Agrawal, H. B. Sailor, M. H. Soni and H. A. Patil,
    "Novel TEO-based Gammatone features for environmental sound classification,"
    2017 25th European Signal Processing Conference (EUSIPCO), Kos, Greece, 2017, pp. 1809-1813,
    doi: 10.23919/EUSIPCO.2017.8081521.

    This is a modified version of the Piczak model, proposed in the paper.

    Since source code was not provided, this is purely constructed by hand,
    with a certain degree of heuristics being applied where unclear.
    """

    def __init__(self, class_count: int, input_channels=2):
        super(TEOGammmatoneCNN, self).__init__()
        self.class_count = class_count
        self.input_size = (60, 41)
        self.fc_in_size = 80 * 1 * 4

        self.main = nn.Sequential(
            # not using in-place operations for now because it might mess with backprop...
            nn.Sequential(
                # 2 @ 60 x 41 / 101
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=80,
                    kernel_size=(57, 6),  # 60 x 6 specified, assuming to be mistake...
                ),
                nn.ReLU(),
                # 80 @ 4 x 36 / 96
                nn.MaxPool2d(kernel_size=(4, 3), stride=(1, 3)),
                # 80 @ 1 x 12 / 32
                nn.Dropout(p=0.5),
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=80, out_channels=80, kernel_size=(1, 3)),
                nn.ReLU(),
                # 80 @ 1 x 10 / 30
                # See https://github.com/lisa-lab/pylearn2/blob/af81e5c362f0df4df85c3e54e23b2adeec026055/pylearn2/models/mlp.py#L3540
                nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3), ceil_mode=True),
                # 80 @ 1 x 4 / 10
                # No Dropout
            ),
            nn.Flatten(start_dim=1),
            nn.Sequential(
                nn.Linear(in_features=self.fc_in_size, out_features=500),
                nn.ReLU(),
                nn.Dropout(p=0.5),
            ),
            nn.Sequential(
                nn.Linear(500, class_count),
                # See https://github.com/lisa-lab/pylearn2/blob/af81e5c362f0df4df85c3e54e23b2adeec026055/pylearn2/models/mlp.py#L1416
                nn.LogSoftmax(dim=1),
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class PiczakCNNBaseline(nn.Module):
    """ESC-50 CNN Baseline

    K. J. Piczak, “Environmental sound classification with convolutional
    neural networks,” in 25th Int. Workshop on Machine Learning for Signal
    Processing (MLSP), Boston, MA, USA, 2015, pp. 1-6.

    Original Repo: https://github.com/karolpiczak/paper-2015-esc-convnet

    Had to rewrite from scratch because the author used `pylearn2` :(
    """

    def __init__(self, class_count: int, long: bool, input_channels=2):
        super(PiczakCNNBaseline, self).__init__()
        self.class_count = class_count
        self.long = long
        self.input_size = (60, 101) if long else (60, 41)

        if long:
            self.fc_in_size = 80 * 1 * 10
        else:
            self.fc_in_size = 80 * 1 * 4

        self.main = nn.Sequential(
            # not using in-place operations for now because it might mess with backprop...
            nn.Sequential(
                # 2 @ 60 x 41 / 101
                nn.Conv2d(
                    in_channels=input_channels, out_channels=80, kernel_size=(57, 6)
                ),
                nn.ReLU(),
                # 80 @ 4 x 36 / 96
                nn.MaxPool2d(kernel_size=(4, 3), stride=(1, 3)),
                # 80 @ 1 x 12 / 32
                nn.Dropout(p=0.5),
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=80, out_channels=80, kernel_size=(1, 3)),
                nn.ReLU(),
                # 80 @ 1 x 10 / 30
                # See https://github.com/lisa-lab/pylearn2/blob/af81e5c362f0df4df85c3e54e23b2adeec026055/pylearn2/models/mlp.py#L3540
                nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3), ceil_mode=True),
                # 80 @ 1 x 4 / 10
                # No Dropout
            ),
            nn.Flatten(start_dim=1),
            nn.Sequential(
                nn.Linear(in_features=self.fc_in_size, out_features=5000),
                nn.ReLU(),
                nn.Dropout(p=0.5),
            ),
            nn.Sequential(
                nn.Linear(in_features=5000, out_features=5000),
                nn.ReLU(),
                nn.Dropout(p=0.5),
            ),
            nn.Sequential(
                nn.Linear(5000, class_count),
                # See https://github.com/lisa-lab/pylearn2/blob/af81e5c362f0df4df85c3e54e23b2adeec026055/pylearn2/models/mlp.py#L1416
                nn.LogSoftmax(dim=1),
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class OurNet(nn.Module):
    """Based on ESC-50 CNN Baseline by K. J. Piczak"""
    
    def __init__(self, class_count: int):
        super(OurNet, self).__init__()
        self.class_count = class_count
        self.input_size = (60, 41)

        self.fc_in_size = 80 * 1 * 4

        self.cnn_along_time = nn.Sequential(
            # not using in-place operations for now because it might mess with backprop...
            nn.Sequential(
                # 2 @ 60 x 41 / 101
                nn.Conv2d(
                    in_channels=2, out_channels=80, kernel_size=(57, 6)
                ),
                nn.ReLU(),
                # 80 @ 4 x 36 / 96
                nn.MaxPool2d(kernel_size=(4, 3), stride=(1, 3)),
                # 80 @ 1 x 12 / 32
                nn.Dropout(p=0.5),
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=80, out_channels=80, kernel_size=(1, 3)),
                nn.ReLU(),
                # 80 @ 1 x 10 / 30
                nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3), ceil_mode=True),
                # 40 @ 1 x 4 / 10
                # nn.Dropout(p=0.5),
            ),
        )

        self.cnn_square_features = nn.Sequential(
            nn.Sequential(
                # 2 @ 60 x 41
                nn.Conv2d(
                    in_channels=2, out_channels=80, kernel_size=(7, 6)
                ),
                nn.BatchNorm2d(80),
                nn.ReLU(),
                # 80 @ 54 x 36
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                # 80 @ 27 x 18
                nn.Dropout(p=0.5),
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=80, out_channels=86, kernel_size=(4, 5)),
                nn.ReLU(),
                # 80 @ 24 x 14
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                # 80 @ 12 x 7
                nn.Dropout(p=0.3),
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=86, out_channels=120, kernel_size=(5, 6)),
                nn.ReLU(),
                # 120 @ 8 x 2
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                # 120 @ 4 x 1
            ),
        )

        self.fc = nn.Sequential(
            nn.Sequential(
                nn.Linear(in_features=int(self.fc_in_size * 2.5), out_features=4096),
                nn.ReLU(),
                nn.Dropout(p=0.5),
            ),
            nn.Sequential(
                nn.Linear(4096, class_count),
                nn.LogSoftmax(dim=1),
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn1_out = torch.flatten(self.cnn_along_time(x), start_dim=1)
        cnn2_out = torch.flatten(self.cnn_square_features(x), start_dim=1)
        cnn_combined = torch.cat((cnn1_out, cnn2_out), dim=1)
        return self.fc(cnn_combined)


class OurCRNNet(nn.Module):
    """Based on ESC-50 CNN Baseline by K. J. Piczak"""

    def __init__(self, class_count: int):
        super(OurCRNNet, self).__init__()
        self.class_count = class_count
        self.input_size = (60, 41)

        self.fc_in_size = 80 * 1 * 4

        self.cnn = nn.Sequential(
            # not using in-place operations for now because it might mess with backprop...
            nn.Sequential(
                # 2 @ 60 x 41 / 101
                nn.Conv2d(
                    in_channels=2, out_channels=80, kernel_size=(57, 6)
                ),
                nn.ReLU(),
                # 80 @ 4 x 36 / 96
                nn.MaxPool2d(kernel_size=(4, 3), stride=(1, 3)),
                # 80 @ 1 x 12 / 32
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=80, out_channels=80, kernel_size=(1, 3)),
                nn.ReLU(),
                # 80 @ 1 x 10 / 30
                nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3), ceil_mode=True),
                # 80 @ 1 x 4 / 10
            ),
            nn.Flatten(start_dim=1),
            nn.Sequential(
                nn.Linear(in_features=self.fc_in_size, out_features=5000),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Linear(in_features=5000, out_features=5000),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Linear(5000, class_count),
            ),
        )
        self.gru = nn.GRU(input_size=class_count, hidden_size=class_count, batch_first=True)
        # self.dropout = nn.Dropout(p=0.25)
        self.output_layer = nn.Sequential(
            nn.Linear(class_count * 2, 1),
            # nn.LogSoftmax(dim=0),  # only one dimesion here
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # alpha = 0.6
        cnn_out: torch.Tensor = self.cnn(x)
        # return torch.max(torch.log_softmax(cnn_out, dim=1), dim=0).values # yields 82%
        gru_out, _ = self.gru(torch.cat([cnn_out, torch.zeros(1, self.class_count, dtype=cnn_out.dtype, device=cnn_out.device)]))
        return torch.log_softmax(gru_out[-1], dim=0)
        # rnn_out = torch.softmax(self.output_layer(gru_out), dim=0) # shape: (seq len, 1)
        # prob_vote = torch.sum(cnn_out * rnn_out, dim=0) # trainable weighted mean
        # return rnn_out * alpha + prob_vote * (1 - alpha)
        # return torch.log_softmax(prob_vote, dim=0) 

    def use_pretrained_piczak(self, state_dict: dict) -> None:
        partial_state_dict = {}
        for k, v in state_dict.items():
            if not str(k).startswith("main."):
                continue
            k2 = str(k).removeprefix("main.")
            if int(k2.split(".")[0]) in {0, 1, 3, 4, 5}:
                partial_state_dict[k2] = v
        # self.cnn.load_state_dict(partial_state_dict)
        self.cnn.load_state_dict(partial_state_dict)
        for name, param in self.cnn.named_parameters():
            if True: # not name.startswith("5"): # Leave ouput layer
                param.requires_grad = False  # Freeze

    @staticmethod
    def collate_fn(data: list[dict]) -> dict[str, list | torch.Tensor]:
        stackable = {
            k: [] for k in data[0].keys() if k not in {"audio", "ranges"} and torch.is_tensor(data[0][k])
        }
        as_is = {
            k: [] for k in data[0].keys() if k not in stackable
        }

        for entry in data:
            for k, v in entry.items():
                if k in stackable:
                    stackable[k].append(v)
                else:
                    as_is[k].append(v)

        # for k, v in { k: v for k, v in stackable.items() }.items():
        #     print(k, [w.shape for w in v])

        return {
            **as_is,
            **{ k: torch.stack(v) for k, v in stackable.items() }
        }


def setup() -> None:
    """Perform setup"""

    config.print()
    print()

    if not os.path.exists(config.cache_dir):
        os.makedirs(config.cache_dir)

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def plot_learning_curve(
    train: list[float], test: list[float], title: str, y_label: str
):
    plt.figure(figsize=(12, 8))
    plt.title(title)
    plt.plot(test, label="Validation")
    plt.plot(train, label="Training")
    plt.xlabel("Epoch Number")
    plt.ylabel(y_label)
    plt.legend()

InputType: TypeAlias = Literal["simple", "ensemble", "no-batch"]

def evaluate(
    net: nn.Module,
    test_loader: DataLoader[dict],
    input_type: InputType,
    compute_conf_matrix: bool,
) -> tuple[float, dict, np.ndarray | None]:
    net.eval()

    test_loss_sum = 0.0
    correct_count = 0
    total_count = 0
    conf_matrix = None

    def evaluate_minibatch(output: torch.Tensor, target: torch.Tensor) -> None:
        nonlocal test_loss_sum
        nonlocal correct_count
        nonlocal total_count
        nonlocal conf_matrix

        class_count = output.size(1)

        # Sum up batch loss
        test_loss_sum += config.criterion(output, target).item()

        # Determine index with maximal log-probability
        predicted = output.argmax(dim=1)
        correct_count += (predicted == target).sum().item()
        total_count += predicted.size(0)

        if compute_conf_matrix:
            # Compute confusion matrix
            current_conf_matrix = metrics.confusion_matrix(
                target.cpu(), predicted.cpu(), labels=list(range(class_count))
            )

            # Update confusion matrix
            if conf_matrix is None:
                conf_matrix = current_conf_matrix
            else:
                conf_matrix += current_conf_matrix

    with torch.no_grad():
        if input_type == "ensemble" or input_type == "no-batch":
            assert test_loader.batch_size is not None
            batch_size = test_loader.batch_size
            dataset_size = len(cast(Sized, test_loader.dataset))

            # For each sample, pass all segments through net and take the average.
            # Then we reconstruct a minibatch output of size (batch size, class count).
            minibatch_output: list[torch.Tensor] = []
            minibatch_target: list[torch.Tensor] = []
            for i in range(dataset_size):
                entry = test_loader.dataset[i]
                segments: torch.Tensor = entry["audio"].to(config.device)
                target: torch.Tensor = entry["target"].to(config.device)

                output: torch.Tensor
                if input_type == "ensemble":
                    # Probability Voting -- take the average of output over all segments
                    segment_output = net(segments)
                    output = torch.mean(segment_output, dim=0)
                else:
                    output = net(segments)
                    assert output.dim() == 1

                minibatch_output.append(output)
                minibatch_target.append(target)

                if (i + 1) % batch_size == 0 or i + 1 == dataset_size:
                    # print(f"Evatuation at minibatch #{i + 1}: ", end="")
                    # print("current minibatch size =", len(minibatch_target))
                    # print(segment_output)
                    # print(output)
                    evaluate_minibatch(
                        torch.stack(minibatch_output), torch.stack(minibatch_target)
                    )

                    minibatch_output = []
                    minibatch_target = []
        else:
            assert input_type == "simple"
            for entry in test_loader:
                data = entry["audio"].to(config.device)
                target: torch.Tensor = entry["target"].to(config.device)

                output = net(data)
                evaluate_minibatch(output, target)

    avg_loss = test_loss_sum / len(test_loader)
    # conf_matrix /= len(test_loader)  # take average

    return (
        avg_loss,
        {
            "correct": correct_count,
            "total": total_count,
            "percentage": 100.0 * correct_count / total_count,
        },
        conf_matrix,
    )


def train(
    net: nn.Module,
    train_loader: DataLoader[dict],
    test_loader: DataLoader[dict],
    optimizer: torch.optim.Optimizer,
    input_type: InputType,
    starting_epoch=1,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None,
) -> tuple[int, dict]:

    print()
    model_info = torchinfo.summary(net, input_size=(config.batch_size, *config.input_shape))
    with open(os.path.join(config.output_dir, "torchinfo.txt"), "w", encoding="utf-8") as model_info_file:
        model_info_file.write(repr(model_info))
    print()

    train_accuracies: list[float] = []
    train_losses: list[float] = []

    test_accuracies: list[float] = []
    test_losses: list[float] = []

    epoch = starting_epoch
    try:
        for epoch in range(starting_epoch, starting_epoch + config.epochs):
            # Epoch summary heading
            print(f"Epoch: {epoch}")

            net.train()

            # Train
            epoch_time_start = time.perf_counter()

            train_loss_sum = 0.0
            train_correct_count = 0
            train_total_count = 0
            minibatch_count = 0
            for batch_index, entry in enumerate(train_loader):
                data: torch.Tensor = entry["audio"]
                target: torch.Tensor = entry["target"]

                target = target.to(config.device)

                if input_type == "no-batch":
                    outputs = []
                    # assert data.dim() == 5
                    # assert target.dim() == 1
                    for i in range(target.size(0)):
                        outputs.append(net(data[i].to(config.device)))
                    output = torch.stack(outputs)
                else:
                    data = data.to(config.device)
                    output: torch.Tensor = net(data)

                # NOTE: loss is actually the mean loss for this mini-batch
                loss: torch.Tensor = config.criterion(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    train_loss_sum += loss.item()

                    # Determine index with maximal log-probability
                    predicted = output.argmax(dim=1)

                    correct_count = (predicted == target).sum().item()

                    train_correct_count += correct_count
                    train_total_count += predicted.size(0)

                    if batch_index % config.report_interval == 0:
                        print(
                            f"\t  > Mini-batch #{batch_index}: \tLoss = {loss.item():.4f} "
                            + f"\tAccuracy = {correct_count} / {predicted.size(0)} "
                            + f"({correct_count / predicted.size(0) * 100:.2f}%)"
                        )

                minibatch_count += 1

            epoch_time_end = time.perf_counter()
            epoch_time = epoch_time_end - epoch_time_start

            # Evaluate
            eval_time_start = time.perf_counter()

            eval_avg_loss, eval_acc, _ = evaluate(
                net, test_loader, input_type, compute_conf_matrix=False
            )

            eval_time_end = time.perf_counter()
            eval_time = eval_time_end - eval_time_start

            train_avg_loss = train_loss_sum / minibatch_count
            train_accuracy_percentage = train_correct_count / train_total_count * 100

            train_accuracies.append(train_accuracy_percentage)
            train_losses.append(train_avg_loss)

            test_accuracies.append(eval_acc["percentage"])
            test_losses.append(eval_avg_loss)

            if scheduler is not None:
                scheduler.step(eval_avg_loss)

            print()
            print(
                f"\t[Train] Time: {epoch_time:.1f}\t| Average Loss: {train_avg_loss:.4f}\t| "
                + f"Accuracy: {train_correct_count}/{train_total_count} ({train_accuracy_percentage:.3f}%)"
                + (f"\t LR = {scheduler.get_last_lr()}" if scheduler is not None else "")
            )
            print(
                f"\t[Test]  Time: {eval_time:.2f}\t| Average Loss: {eval_avg_loss:.4f}\t| "
                + f"Accuracy: {eval_acc['correct']}/{eval_acc['total']} ({eval_acc['percentage']:.2f}%)"
            )
            print()

            if (
                config.halt_accuracy_percentage is not None
                and eval_acc['percentage'] >= config.halt_accuracy_percentage
            ):
                break

            if train_avg_loss < 0.001 or train_correct_count == train_total_count:
                break

    except KeyboardInterrupt:
        epoch -= 1  # Incomplete epoch
        print("[Keyboard Interrupt]")

    print()
    print("Training complete.")
    print()

    np.save(os.path.join(config.output_dir, "accuracy_train.npz"), train_accuracies)
    np.save(os.path.join(config.output_dir, "accuracy_test.npz"), test_accuracies)

    plot_learning_curve(train_accuracies, test_accuracies, "Accuracy", "Accuracy (%)")
    plt.savefig(os.path.join(config.output_dir, "plot_accuracy.png"))
    if config.display_graphics:
        plt.show()

    np.save(os.path.join(config.output_dir, "loss_train.npz"), train_accuracies)
    np.save(os.path.join(config.output_dir, "loss_test.npz"), test_accuracies)

    plot_learning_curve(train_losses, test_losses, "Loss", config.criterion_name)
    plt.savefig(os.path.join(config.output_dir, "plot_loss.png"))
    if config.display_graphics:
        plt.show()

    return epoch, optimizer.state_dict()


def test(net: nn.Module, test_loader: DataLoader[dict], input_type:InputType) -> None:
    """For testing outside training."""
    # Evaluate
    eval_time_start = time.perf_counter()

    eval_avg_loss, eval_acc, conf_matrix = evaluate(
        net, test_loader, input_type, compute_conf_matrix=True
    )

    eval_time_end = time.perf_counter()
    eval_time = eval_time_end - eval_time_start

    print(
        f"[Eval]  Time: {eval_time:.2f}\t| Average Loss: {eval_avg_loss:.4f}\t| "
        + f"Accuracy: {eval_acc['correct']}/{eval_acc['total']} ({eval_acc['percentage']:.3f}%)"
    )

    assert conf_matrix is not None
    np.savetxt(os.path.join(config.output_dir, "confusion_matrix.txt"), conf_matrix)

    print("> Confusion Matrix:")
    with np.printoptions(precision=4, threshold=sys.maxsize, linewidth=256):
        print(conf_matrix)

    full_fig_size = conf_matrix.shape[0] * 0.8 + 0.5
    plt.figure(figsize=(full_fig_size, full_fig_size))
    metrics.ConfusionMatrixDisplay(conf_matrix).plot(ax=plt.gca())
    plt.savefig(os.path.join(config.output_dir, "confusion_matrix.png"))

    if config.display_graphics:
        plt.show()


def train_and_test(dataset: RawDataset, model: nn.Module, input_type: InputType) -> None:
    # pipeline = OurPipeline(dataset.get_sample_rate()).to(config.device).eval()

    train_dataset, test_dataset = dataset.get_train_and_test_datasets(
        fold_count=config.fold_count,
        test_fold_index=config.test_fold_index,
    )

    print(
        "The size of an example dataset entry size is",
        np.shape(test_dataset[1]["audio"]),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=False #, collate_fn=OurCRNNet.collate_fn
    )
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        nesterov=config.nesterov_momentum,
    )

    # For continuing training
    scheduler = None  # torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    starting_epoch = 1
    optimizer_state = None
    if config.model_path is not None:
        saved: dict = torch.load(config.model_path)
        starting_epoch = saved["epoch_count"] + 1
        optimizer.load_state_dict(saved["optimizer"])
        model.load_state_dict(saved["state_dict"])
        print(f"Loaded existing model at '{config.model_path}'.")
        # test(model, test_loader, True);
        # exit()
    else:
        # print([(n, p.mean().item()) for n, p in model.named_parameters()])
        init_weights(model)
        # print([(n, p.mean().item()) for n, p in model.named_parameters()])
        # exit()

    # model.use_pretrained_piczak(torch.load("output/esc10-pure-teo-gtsc/570/model.pt")["state_dict"])
    # test(model, test_loader, input_type)

    epoch_count, optimizer_state = train(
        model,
        train_loader,
        test_loader,
        optimizer,
        input_type,
        starting_epoch=starting_epoch,
        scheduler=scheduler
    )

    test(model, test_loader, input_type)

    torch.save(
        {
            "epoch_count": epoch_count,
            "state_dict": model.state_dict(),
            "optimizer": optimizer_state,
        },
        os.path.join(config.output_dir, "model.pt"),
    )


def main() -> None:
    # config.input_shape = (1, 224, 224) # VGG-16
    config.input_shape = (2, 60, 41)  # Piczak CNN, Short Segments
    # config.input_shape = (2, 60, 101)  # Piczak CNN, Long Segments

    # config.input_shape = (2, 60, 41)  # Our CRNN, Short Segments
    # config.batch_size = 25 # For CRNN

    config.epochs = 600

    setup()
    
    # dataset = DebugDataset(4)
    # dataset = ESCFullMelDataset(esc10=True)
    # dataset = ESCLogMelSegmentDataset(esc10=True, long=True, augment_count=0)
    # dataset = ESCLogMelSegmentDataset(esc10=True, long=False, augment_count=0)
    # dataset = ESCLogMelSegmentDataset(esc10=True, long=False, augment_count=0, use_test_format=True)
    # dataset = ESCAugmentedLogMelSegmentDataset(esc10=True, long=False)
    dataset = EscFusedTeoGtscDataset(esc10=True)
    # dataset = OurDataset(esc10=True, long=False, use_test_format=False)
    # dataset = OurDataset(esc10=True, long=False, use_test_format=True)

    # model = DebugNet(dataset.get_class_count())
    # model = VGG16(dataset.get_class_count())
    # model = PiczakCNNBaseline(dataset.get_class_count(), long=False)
    # model = PiczakCNNBaseline(dataset.get_class_count(), long=False)
    # model = TEOGammmatoneCNN(dataset.get_class_count())
    model = OurNet(dataset.get_class_count())
    # model = OurCRNNet(dataset.get_class_count())
    model = model.to(config.device)

    input_type: InputType = "ensemble"
    # input_type: InputType = "no-batch"

    config.halt_accuracy_percentage = 98

    train_and_test(dataset, model, input_type)


if __name__ == "__main__":
    main()
