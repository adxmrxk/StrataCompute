"""Measure StrataMotion's drift guard against normal and corrupted raw windows."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo.har_features import apply_linear_adapter, extract_uci_features

DATASET = ROOT / "data" / "uci_har" / "UCI HAR Dataset"
ARTIFACTS = ROOT / "data" / "demo"


def load_signals() -> tuple[np.ndarray, np.ndarray]:
    signals = DATASET / "test" / "Inertial Signals"
    total = np.stack([
        np.loadtxt(signals / f"total_acc_{axis}_test.txt", dtype=np.float64)
        for axis in "xyz"
    ], axis=2)
    gyro = np.stack([
        np.loadtxt(signals / f"body_gyro_{axis}_test.txt", dtype=np.float64)
        for axis in "xyz"
    ], axis=2)
    return total, gyro


def outlier_rate(features: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean((features < lower) | (features > upper)))


def main() -> None:
    adapter = np.load(ARTIFACTS / "feature_adapter.npz")
    health = np.load(ARTIFACTS / "input_health.npz")
    total, gyro = load_signals()
    # Fixed spread of held-out windows, not cherry-picked examples.
    indices = np.linspace(0, total.shape[0] - 1, 60, dtype=int)
    scenarios = {
        "normal": (1.0, 1.0),
        "accelerometer_unit_x10": (10.0, 1.0),
        "gyroscope_unit_x20": (1.0, 20.0),
        "combined_unit_error": (10.0, 20.0),
    }
    results = {}
    for name, (accel_scale, gyro_scale) in scenarios.items():
        rates = []
        sanity_reviews = []
        for index in indices:
            acceleration = total[index] * accel_scale
            gyroscope = gyro[index] * gyro_scale
            raw = extract_uci_features(acceleration, gyroscope)
            features = apply_linear_adapter(raw, adapter["mapping"], adapter["offset"])
            rates.append(outlier_rate(features, health["lower"], health["upper"]))
            sanity_reviews.append(
                np.linalg.norm(acceleration, axis=1).max() > 4.0 or
                np.linalg.norm(gyroscope, axis=1).max() > 6.0
            )
        detected = np.asarray(rates) > float(health["threshold"])
        results[name] = {
            "windows": int(indices.size),
            "mean_outlier_fraction": float(np.mean(rates)),
            "review_rate": float(np.mean(detected)),
            "combined_review_rate": float(np.mean(detected | np.asarray(sanity_reviews))),
        }
    print("Drift guard evaluation (60 evenly spaced UCI held-out raw windows per scenario)")
    print(f"Outlier budget: {float(health['threshold']):.2%}")
    for name, result in results.items():
        print(f"{name}: mean_outliers={result['mean_outlier_fraction']:.2%}, "
              f"drift_review={result['review_rate']:.1%}, "
              f"combined_review={result['combined_review_rate']:.1%} ({result['windows']} windows)")
    corrupted = [results[key]["combined_review_rate"] for key in scenarios if key != "normal"]
    print(f"Baseline without a detector: 0.0% unit/scale corruptions detected")
    print(f"Current detector: {float(np.mean(corrupted)):.1%} mean detection across 3 corruption scenarios")


if __name__ == "__main__":
    main()
