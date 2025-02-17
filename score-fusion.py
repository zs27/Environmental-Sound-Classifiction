from argparse import ArgumentParser
import os
import sys
import time
from typing import Sized, cast

from matplotlib import pyplot as plt
from sklearn import metrics
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader

from train import config, test, OurNet, EscPureTeoGtscDataset, EscLogMelSegmentDataset

def evaluate2(
    net1: nn.Module,
    net2: nn.Module,
    alpha: float,
    test_loader1: DataLoader[dict],
    test_loader2: DataLoader[dict],
    ensemble: bool,
    compute_conf_matrix: bool,
) -> tuple[float, dict, np.ndarray | None]:
    net1.eval()
    net2.eval()

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
        assert ensemble
        assert test_loader1.batch_size is not None
        assert test_loader1.batch_size == test_loader2.batch_size
        assert len(test_loader1) == len(test_loader2)
        batch_size = test_loader1.batch_size
        dataset_size = len(cast(Sized, test_loader1.dataset))
        assert dataset_size == len(cast(Sized, test_loader2.dataset))

        # For each sample, pass all segments through net and take the average.
        # Then we reconstruct a minibatch output of size (batch size, class count).
        minibatch_output: list[torch.Tensor] = []
        minibatch_target: list[torch.Tensor] = []
        for i in range(dataset_size):
            entry1 = test_loader1.dataset[i]
            entry2 = test_loader2.dataset[i]
            
            segments1: torch.Tensor = entry1["audio"].to(config.device)
            segments2: torch.Tensor = entry2["audio"].to(config.device)
            # assert segments1.size(0) == segments2.size(0)

            target: torch.Tensor = entry1["target"].to(config.device)
            assert target.item() == entry2["target"].item()

            # Probability Voting -- take the average of output over all segments
            segment_output1: torch.Tensor = net1(segments1)
            segment_output2: torch.Tensor = net2(segments2)

            # print(segment_output1.size(), segment_output2.size())
            # assert segment_output1.size() == segment_output2.size()
            if True or segment_output1.size() != segment_output2.size():
                segment_output1 = torch.mean(segment_output1, dim=0)
                segment_output2 = torch.mean(segment_output2, dim=0)

                segment_output = segment_output1 * alpha + segment_output2 * (1 - alpha)
                segment_output -= 0.1 * segment_output1 * segment_output2
            
                output = segment_output
            else:
                segment_output = segment_output1 * segment_output2
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

    avg_loss = test_loss_sum / len(test_loader1)
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

def test2(net1: nn.Module, net2: nn.Module, alpha: float, test_loader1: DataLoader[dict], test_loader2: DataLoader[dict], ensemble: bool) -> None:
    """For testing outside training."""
    # Evaluate
    eval_time_start = time.perf_counter()

    eval_avg_loss, eval_acc, conf_matrix = evaluate2(
        net1, net2, alpha, test_loader1, test_loader2, ensemble, compute_conf_matrix=True
    )

    eval_time_end = time.perf_counter()
    eval_time = eval_time_end - eval_time_start

    print(
        f"[Eval]  Time: {eval_time:.2f}\t| Average Loss: {eval_avg_loss:.4f}\t| "
        + f"Accuracy: {eval_acc['correct']}/{eval_acc['total']} ({eval_acc['percentage']:.0f}%)"
    )

    assert conf_matrix is not None
    np.savetxt(os.path.join(".", "confusion_matrix.txt"), conf_matrix)

    print("> Confusion Matrix:")
    with np.printoptions(precision=4, threshold=sys.maxsize, linewidth=256):
        print(conf_matrix)

    full_fig_size = conf_matrix.shape[0] * 0.8 + 0.5
    plt.figure(figsize=(full_fig_size, full_fig_size))
    metrics.ConfusionMatrixDisplay(conf_matrix).plot(ax=plt.gca())
    plt.savefig(os.path.join(".", "confusion_matrix.png"))

    if config.display_graphics:
        plt.show()

class Fused(nn.Module):
    def __init__(self, a: nn.Module, b: nn.Module, alpha: float) -> None:
        super().__init__()
        self.a = a
        self.b = b
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.a(x) + (1 - self.alpha) * self.b(x)

def load_model(path: str, model: OurNet) -> OurNet:
    saved: dict = torch.load(path)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    print(f"Loaded existing model at '{path}'.")
    return model

def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("-a", "--alpha", type=float, default=0.5)
    parser.add_argument("model1", help="ours")
    parser.add_argument("model2", help="log-mel")

    opts = parser.parse_args()

    config.output_dir = "."
    config.display_graphics = False

    dataset1 = EscPureTeoGtscDataset(esc10=True, long=False)
    dataset2 = EscLogMelSegmentDataset(esc10=True, long=False)

    _, test1_dataset = dataset1.get_train_and_test_datasets(
        fold_count=config.fold_count,
        test_fold_index=config.test_fold_index,
    )

    _, test2_dataset = dataset2.get_train_and_test_datasets(
        fold_count=config.fold_count,
        test_fold_index=config.test_fold_index,
    )

    test1_loader = DataLoader(test1_dataset, batch_size=config.batch_size, shuffle=False)
    test2_loader = DataLoader(test2_dataset, batch_size=config.batch_size, shuffle=False)

    model1 = load_model(opts.model1, OurNet(dataset1.get_class_count()).to(config.device))
    model2 = load_model(opts.model2, OurNet(dataset2.get_class_count()).to(config.device))

    test(model1, test1_loader, "ensemble")
    test(model2, test2_loader, "ensemble")

    test2(model1, model2, opts.alpha, test1_loader, test2_loader, ensemble=True)

if __name__ == "__main__":
    main()
