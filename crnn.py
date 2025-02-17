import multiprocessing
import os
import sys
import time
from abc import ABC, abstractmethod
from pprint import pprint
from typing import Iterator, Literal, Sized, TypeAlias, cast

import datasets
import datasets.config
import gammatone.filters
import gammatone.gtgram
import librosa
import matplotlib.pyplot as plt
import nnAudio.features
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils
import torchaudio.transforms as T
import torchinfo
from gammatone.filters import centre_freqs, erb_filterbank, make_erb_filters
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info

from train import config, setup

class OurCRNNet(nn.Module):
    """Based on ESC-50 CNN Baseline by K. J. Piczak"""

    def __init__(self, class_count: int):
        super(OurCRNNet, self).__init__()
        self.class_count = 2
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
                nn.Dropout(p=0.5),
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=80, out_channels=80, kernel_size=(1, 3)),
                nn.ReLU(),
                # 80 @ 1 x 10 / 30
                nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3), ceil_mode=True),
                # 80 @ 1 x 4 / 10
                # nn.Dropout(p=0.5),
            )
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.gru = nn.GRU(input_size=self.fc_in_size, hidden_size=256)
        self.output_layer = nn.Sequential(
            nn.Linear(256, class_count),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn_out = self.cnn(x)
        flattened = self.flatten(cnn_out)
        gru_out, _ = self.gru(flattened)
        return self.output_layer(gru_out[-1])


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


def evaluate(
    net: nn.Module,
    test_loader: DataLoader[dict],
    ensemble: bool,
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
        if ensemble:
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

                # Probability Voting -- take the average of output over all segments
                segment_output = net(segments)
                output = torch.mean(segment_output, dim=0)

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
    ensemble: bool,
    starting_epoch=1,
) -> tuple[int, dict]:

    print()
    torchinfo.summary(net, input_size=(config.batch_size, *config.input_shape))
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

                data = data.to(config.device)
                target = target.to(config.device)

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
                            + f"({correct_count / predicted.size(0) * 100:.0f}%)"
                        )

                minibatch_count += 1

            epoch_time_end = time.perf_counter()
            epoch_time = epoch_time_end - epoch_time_start

            # Evaluate
            eval_time_start = time.perf_counter()

            eval_avg_loss, eval_acc, _ = evaluate(
                net, test_loader, ensemble, compute_conf_matrix=False
            )

            eval_time_end = time.perf_counter()
            eval_time = eval_time_end - eval_time_start

            train_avg_loss = train_loss_sum / minibatch_count
            train_accuracy_percentage = train_correct_count / train_total_count * 100

            train_accuracies.append(train_accuracy_percentage)
            train_losses.append(train_avg_loss)

            test_accuracies.append(eval_acc["percentage"])
            test_losses.append(eval_avg_loss)

            print()
            print(
                f"\t[Train] Time: {epoch_time:.1f}\t| Average Loss: {train_avg_loss:.4f}\t| "
                + f"Accuracy: {train_correct_count}/{train_total_count} ({train_accuracy_percentage:.0f}%)"
            )
            print(
                f"\t[Test]  Time: {eval_time:.2f}\t| Average Loss: {eval_avg_loss:.4f}\t| "
                + f"Accuracy: {eval_acc['correct']}/{eval_acc['total']} ({eval_acc['percentage']:.0f}%)"
            )
            print()

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


def test(net: nn.Module, test_loader: DataLoader[dict], ensemble: bool) -> None:
    """For testing outside training."""
    # Evaluate
    eval_time_start = time.perf_counter()

    eval_avg_loss, eval_acc, conf_matrix = evaluate(
        net, test_loader, ensemble, compute_conf_matrix=True
    )

    eval_time_end = time.perf_counter()
    eval_time = eval_time_end - eval_time_start

    print(
        f"[Eval]  Time: {eval_time:.2f}\t| Average Loss: {eval_avg_loss:.4f}\t| "
        + f"Accuracy: {eval_acc['correct']}/{eval_acc['total']} ({eval_acc['percentage']:.0f}%)"
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


def main() -> None:
    # config.input_shape = (1, 224, 224) # VGG-16
    config.input_shape = (2, 60, 41)  # Piczak CNN, Short Segments
    # config.input_shape = (2, 60, 101)  # Piczak CNN, Long Segments

    setup()

    # dataset = DebugDataset(4)
    # dataset = ESCFullMelDataset(esc10=True)
    # dataset = ESCLogMelSegmentDataset(esc10=True, long=True, augment_count=0)
    # dataset = ESCLogMelSegmentDataset(esc10=True, long=False, augment_count=0)
    # dataset = ESCAugmentedLogMelSegmentDataset(esc10=True, long=False)
    # dataset = ESCGammatoneTEODataset(esc10=True)
    dataset = OurDataset(esc10=True, long=False)

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
        train_dataset, batch_size=config.batch_size, shuffle=False
    )
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    ensemble = True
    # model = DebugNet(dataset.get_class_count())
    # model = VGG16(dataset.get_class_count())
    # model = PiczakCNNBaseline(dataset.get_class_count(), long=False)
    model = PiczakCNNBaseline(dataset.get_class_count(), long=False)
    # model = TEOGammmatoneCNN(dataset.get_class_count())
    # model = OurNet(dataset.get_class_count())
    model = model.to(config.device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        nesterov=config.nesterov_momentum,
    )

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


    epoch_count, optimizer_state = train(
        model,
        train_loader,
        test_loader,
        optimizer,
        ensemble=ensemble,
        starting_epoch=starting_epoch,
    )

    test(model, test_loader, ensemble=ensemble)

    torch.save(
        {
            "epoch_count": epoch_count,
            "state_dict": model.state_dict(),
            "optimizer": optimizer_state,
        },
        os.path.join(config.output_dir, "model.pt"),
    )


if __name__ == "__main__":
    main()

