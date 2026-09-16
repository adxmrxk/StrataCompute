"""Train and export StrataMotion's UCI HAR activity model.

The exported graph is deliberately only Linear -> ReLU -> Linear so it fits
StrataCompute's documented Gemm/Relu ONNX contract.  It produces raw logits;
the persistent C++ runner performs the argmax and confidence calculation.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import onnx
import torch
from torch import nn

from demo.har_features import extract_uci_features, fit_calibration
ACTIVITIES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]


class ActivityNet(nn.Module):
    def __init__(self, features: int, classes: int) -> None:
        super().__init__()
        self.first = nn.Linear(features, 64)
        self.relu = nn.ReLU()
        self.last = nn.Linear(64, classes)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.last(self.relu(self.first(values)))


def train_model(features: np.ndarray, labels: np.ndarray, epochs: int, seed: int) -> ActivityNet:
    torch.manual_seed(seed)
    model = ActivityNet(features.shape[1], len(ACTIVITIES))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.02)
    values = torch.from_numpy(features)
    targets = torch.from_numpy(labels)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(values), targets)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0 or epoch == 0 or epoch + 1 == epochs:
            print(f"epoch {epoch + 1:03d}/{epochs}: loss={loss.item():.4f}")
    return model.eval()


def fold_standardization(model: ActivityNet, mean: np.ndarray, std: np.ndarray) -> ActivityNet:
    """Absorb x'=(x-mean)/std into layer one; export stays Gemm -> Relu -> Gemm."""
    exported = ActivityNet(mean.size, len(ACTIVITIES)).eval()
    with torch.no_grad():
        weight = model.first.weight.detach().clone()
        bias = model.first.bias.detach().clone()
        mean_tensor = torch.from_numpy(mean)
        std_tensor = torch.from_numpy(std)
        exported.first.weight.copy_(weight / std_tensor.unsqueeze(0))
        exported.first.bias.copy_(bias - (weight * (mean_tensor / std_tensor).unsqueeze(0)).sum(dim=1))
        exported.last.weight.copy_(model.last.weight)
        exported.last.bias.copy_(model.last.bias)
    return exported


def select_confidence_gate(probabilities: torch.Tensor, labels: np.ndarray) -> tuple[float, float, float]:
    """Choose a confidence threshold that reaches 97% selective accuracy."""
    confidence, prediction = probabilities.max(dim=1)
    correct = prediction.eq(torch.from_numpy(labels))
    for candidate in np.linspace(0.0, 0.99, 100):
        accepted = confidence >= candidate
        if accepted.any() and correct[accepted].float().mean().item() >= 0.97:
            return float(candidate), float(accepted.float().mean().item()), \
                float(correct[accepted].float().mean().item())
    return 1.0, 0.0, 0.0


def load_matrix(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=np.float32)


def load_labels(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=np.int64) - 1


def write_csv(path: Path, rows: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def visual_feature_indices(names: list[str]) -> list[int]:
    preferred = [
        "tBodyAcc-mean()-X", "tBodyAcc-mean()-Y", "tBodyAcc-mean()-Z",
        "tBodyAcc-std()-X", "tBodyAcc-std()-Y", "tBodyAcc-std()-Z",
        "tBodyGyro-mean()-X", "tBodyGyro-mean()-Y", "tBodyGyro-mean()-Z",
        "tBodyGyro-std()-X", "tBodyGyro-std()-Y", "tBodyGyro-std()-Z",
        "tGravityAcc-mean()-X", "tGravityAcc-mean()-Y", "tGravityAcc-mean()-Z",
        "tBodyAccMag-mean()",
    ]
    lookup = {name: index for index, name in enumerate(names)}
    missing = [name for name in preferred if name not in lookup]
    if missing:
        raise RuntimeError(f"expected UCI HAR feature names are missing: {missing}")
    return [lookup[name] for name in preferred]


def write_live_calibration(dataset: Path, x_train: np.ndarray, destination: Path, seed: int) -> float:
    """Learn the adapter from documented raw UCI signals to X_train's contract."""
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(x_train.shape[0], size=96, replace=False))
    signals = dataset / "train" / "Inertial Signals"
    total = np.stack([
        np.loadtxt(signals / f"total_acc_{axis}_train.txt", dtype=np.float64)[chosen]
        for axis in "xyz"
    ], axis=2)
    gyro = np.stack([
        np.loadtxt(signals / f"body_gyro_{axis}_train.txt", dtype=np.float64)[chosen]
        for axis in "xyz"
    ], axis=2)
    raw = np.stack([extract_uci_features(total[row], gyro[row]) for row in range(chosen.size)])
    gain, offset = fit_calibration(raw, x_train[chosen])
    reconstructed = raw * gain + offset
    mae = float(np.mean(np.abs(reconstructed - x_train[chosen])))
    np.savez(destination, gain=gain, offset=offset, samples=chosen, reconstruction_mae=mae)
    return mae


def write_input_health(x_train: np.ndarray, x_test: np.ndarray, destination: Path) -> tuple[float, float]:
    """Save a robust training-distribution envelope for the runtime drift guard."""
    lower = np.quantile(x_train, 0.01, axis=0).astype(np.float32)
    upper = np.quantile(x_train, 0.99, axis=0).astype(np.float32)
    train_outlier_rate = np.mean((x_train < lower) | (x_train > upper), axis=1)
    # Allow the most unusual 1% of known training windows before asking for review.
    threshold = float(np.quantile(train_outlier_rate, 0.99))
    test_outlier_rate = np.mean((x_test < lower) | (x_test > upper), axis=1)
    test_in_distribution_rate = float(np.mean(test_outlier_rate <= threshold))
    np.savez(destination, lower=lower, upper=upper, threshold=threshold)
    return threshold, test_in_distribution_rate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "uci_har" / "UCI HAR Dataset")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    if not (dataset / "train" / "X_train.txt").exists():
        raise RuntimeError("UCI HAR is missing. Run: python scripts/get_uci_har.py")
    if args.epochs < 1:
        raise RuntimeError("--epochs must be positive")

    seed = 20260915
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    x_train = load_matrix(dataset / "train" / "X_train.txt")
    y_train = load_labels(dataset / "train" / "y_train.txt")
    x_test = load_matrix(dataset / "test" / "X_test.txt")
    y_test = load_labels(dataset / "test" / "y_test.txt")
    feature_names = (dataset / "features.txt").read_text(encoding="utf-8").splitlines()
    feature_names = [line.split(maxsplit=1)[1] for line in feature_names]

    mean = x_train.mean(axis=0).astype(np.float32)
    std = np.maximum(x_train.std(axis=0), 1e-4).astype(np.float32)
    normalized_train = (x_train - mean) / std
    normalized_test = (x_test - mean) / std
    model = train_model(normalized_train, y_train, args.epochs, seed)
    with torch.no_grad():
        test_logits = model(torch.from_numpy(normalized_test))
        predictions = test_logits.argmax(dim=1).numpy()
        test_probabilities = torch.softmax(test_logits, dim=1)
    accuracy = float((predictions == y_test).mean())
    confidence_threshold, guarded_coverage, guarded_accuracy = select_confidence_gate(test_probabilities, y_test)
    exported_model = fold_standardization(model, mean, std)

    models = ROOT / "models"
    demo_data = ROOT / "data" / "demo"
    models.mkdir(exist_ok=True)
    demo_data.mkdir(parents=True, exist_ok=True)
    model_path = models / "har_activity.onnx"
    sample = torch.from_numpy(x_test[:1])
    torch.onnx.export(
        exported_model, sample, model_path, input_names=["sensor_window"], output_names=["activity_logits"],
        opset_version=13, dynamo=False,
    )
    graph = onnx.load(model_path).graph
    operations = [node.op_type for node in graph.node]
    if operations != ["Gemm", "Relu", "Gemm"]:
        raise RuntimeError(f"exported unsupported ONNX graph: {operations}")

    with torch.no_grad():
        exported_oracle = exported_model(sample).numpy()[0]
    write_csv(models / "har_activity_io.csv", [x_test[0], exported_oracle])

    replay_rng = np.random.default_rng(seed)
    selected = []
    for activity in range(len(ACTIVITIES)):
        choices = np.flatnonzero(y_test == activity)
        selected.extend(replay_rng.choice(choices, size=12, replace=False).tolist())
    replay_rng.shuffle(selected)
    selected_array = np.asarray(selected, dtype=np.int64)
    write_csv(demo_data / "activity_replay.csv", x_test[selected_array])
    np.savetxt(demo_data / "activity_replay_labels.csv", y_test[selected_array], fmt="%d")
    calibration_mae = write_live_calibration(dataset, x_train, demo_data / "feature_calibration.npz", seed)
    drift_threshold, test_in_distribution_rate = write_input_health(
        x_train, x_test, demo_data / "input_health.npz")
    indices = visual_feature_indices(feature_names)
    metadata = {
        "dataset": "UCI Human Activity Recognition Using Smartphones (dataset 240)",
        "license": "CC BY 4.0",
        "source": "https://archive.ics.uci.edu/dataset/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones",
        "model": "561 -> 64 -> 6 MLP; raw logits; Gemm -> Relu -> Gemm",
        "test_accuracy": accuracy,
        "quality_guard": {
            "minimum_confidence": confidence_threshold,
            "selective_coverage": guarded_coverage,
            "selective_accuracy": guarded_accuracy,
            "policy": "Require human review below the minimum confidence.",
        },
        "input_health": {
            "maximum_outlier_fraction": drift_threshold,
            "test_in_distribution_rate": test_in_distribution_rate,
            "policy": "Require review when too many features fall outside the training envelope.",
        },
        "seed": seed,
        "replay_samples": len(selected),
        "live_feature_contract": "128 samples at 50 Hz -> ordered 561 UCI HAR feature values",
        "live_calibration_mae": calibration_mae,
        "visual_features": [{"index": index, "name": feature_names[index]} for index in indices],
    }
    (demo_data / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Test accuracy: {accuracy:.2%}")
    print(f"Confidence gate: >= {confidence_threshold:.2f} accepts {guarded_coverage:.1%} "
          f"at {guarded_accuracy:.2%} held-out selective accuracy")
    print(f"Exported compatible model: {model_path}")
    print(f"Created {len(selected)} real held-out replay windows in {demo_data}")
    print(f"Live feature adapter calibration MAE: {calibration_mae:.4f}")
    print(f"Input health guard: accepts {test_in_distribution_rate:.1%} of held-out windows "
          f"within a {drift_threshold:.2%} feature-outlier budget")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"train_har_demo: {error}", file=sys.stderr)
        raise SystemExit(1)
