"""Measure compatible MLP candidates before changing StrataMotion's model."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "uci_har" / "UCI HAR Dataset"


class Candidate(nn.Module):
    def __init__(self, widths: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = 561
        for width in widths:
            layers.extend((nn.Linear(previous, width), nn.ReLU()))
            previous = width
        layers.append(nn.Linear(previous, 6))
        self.layers = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


def load(partition: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.loadtxt(DATASET / partition / f"X_{partition}.txt", dtype=np.float32)
    y = np.loadtxt(DATASET / partition / f"y_{partition}.txt", dtype=np.int64) - 1
    return x, y


def run(widths: tuple[int, ...], epochs: int = 100) -> tuple[float, float, float]:
    torch.manual_seed(20260915)
    x_train, y_train = load("train")
    x_test, y_test = load("test")
    mean = x_train.mean(axis=0)
    std = np.maximum(x_train.std(axis=0), 1e-4)
    train = torch.from_numpy((x_train - mean) / std)
    test = torch.from_numpy((x_test - mean) / std)
    target = torch.from_numpy(y_train)
    model = Candidate(widths)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=0.0005)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.02)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(train), target)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(test), dim=1)
        confidence, predicted = probabilities.max(dim=1)
        correct = predicted.eq(torch.from_numpy(y_test))
        accuracy = correct.float().mean().item()
        # Choose the lowest confidence threshold that reaches 97% selective accuracy.
        threshold = 1.0
        coverage = 0.0
        for candidate in np.linspace(0.0, 0.99, 100):
            accepted = confidence >= candidate
            if accepted.any() and correct[accepted].float().mean().item() >= 0.97:
                threshold = float(candidate)
                coverage = accepted.float().mean().item()
                break
    return accuracy, threshold, coverage


def main() -> None:
    torch.set_num_threads(1)
    for widths in ((64,), (96, 48), (128, 64)):
        accuracy, threshold, coverage = run(widths)
        description = " -> ".join(map(str, (561, *widths, 6)))
        print(f"{description}: test_accuracy={accuracy:.2%}, "
              f"97%-accuracy guard: confidence>={threshold:.2f}, coverage={coverage:.1%}")


if __name__ == "__main__":
    main()
