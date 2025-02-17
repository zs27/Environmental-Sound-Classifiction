# Dataset inspection helper

import argparse
import sys
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import re

from train import EscDataset, EscFusedTeoGtscDataset, EscLogMelSegmentDataset, EscPureTeoGtscDataset


def plot_test_entries(dataset: EscDataset, name: re.Pattern) -> None:
    for fold in dataset.test_folds:
        for row in fold:
            row = cast(dict, row)
            if name.match(row["filename"]):
                print("=" * 16, "Found entry:", row["filename"])
                plot_entry(dataset, row)
    print("Traversed all 5 folds.")


def plot_train_entries(dataset: EscDataset, name: re.Pattern) -> None:
    for fold in dataset.train_folds:
        i = 0
        while i < len(fold):
            row = cast(dict, fold[i])
            pseudo_row = {**row, "audio": []}
            while name.match(row["filename"]):
                print("=" * 16, "Found entry:", row["filename"], f"[{i}]")
                pseudo_row["audio"].append(row["audio"])
                i += 1
                row = cast(dict, fold[i])
            if len(pseudo_row["audio"]) > 0:
                plot_entry(dataset, pseudo_row)
            else:
                i += 1
    print("Traversed all 5 folds.")


def plot_entry(dataset: EscDataset, entry: dict) -> None:
    assert entry is not None

    segments = entry["audio"]

    n = len(segments)

    print(n, "segments found.")
    print(entry["target"], entry.get("ranges"))

    plt.figure(figsize=(n + 1, 2 + 1))

    for i, spec in enumerate(segments):
        plt.subplot(2, n, i + 1)
        plt.imshow(spec[0], aspect="auto", origin="lower")
        plt.title("Segment " + str(i + 1))

        plt.subplot(2, n, n + i + 1)
        plt.imshow(spec[1], aspect="auto", origin="lower")
        plt.title("Delta")

        print(
            f"Segment #{i + 1}: min = {np.min(spec[0])}, max = {np.max(spec[0])}, mean = {np.mean(spec[0])}"
        )

    plt.tight_layout()
    plt.show()

    np.savetxt(
        "output/segment-dump-" + dataset.get_name() + ".txt", segments[0][0], "%.4f"
    )
    np.savetxt(
        "output/segment-dump-" + dataset.get_name() + ".delta.txt",
        segments[0][1],
        "%.4f",
    )


def main() -> None:
    esc10 = True
    long = False
    datasets = {
        "log-mel": lambda: EscLogMelSegmentDataset(esc10=esc10, long=long),
        "gtsc": lambda: EscFusedTeoGtscDataset(esc10=esc10),
        "ours": lambda: EscPureTeoGtscDataset(esc10=esc10, long=long)
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, choices=datasets.keys())
    parser.add_argument("type", type=str, choices=["train", "test"])
    parser.add_argument(
        "file",
        help="The filename (not path!) of the raw wav file (regex allowed)",
        type=str,
    )

    opts = parser.parse_args()

    dataset = datasets[opts.dataset]()
    name = re.compile(opts.file)

    print("Regex:", name)

    try:
        (plot_train_entries if opts.type == "train" else plot_test_entries)(
            dataset, name
        )
    except KeyboardInterrupt:
        print("[Ctrl-C]")


if __name__ == "__main__":
    main()
