"""UCI HAR's ordered 561-feature contract for a 2.56 s / 50 Hz motion window.

This is a local preprocessing bridge for live device-motion samples.  It
recreates the documented signal families and feature order used by UCI HAR:
body/gravity acceleration, gyroscope and jerk signals, magnitudes, FFT-domain
statistics, band energies, and seven angle features.  `fit_calibration` learns
a per-feature affine adapter from the downloaded UCI raw windows to the
repository's trained feature matrix, since the public `X_train.txt` values are
the training representation consumed by the exported model.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

SAMPLE_RATE = 50.0
WINDOW_SAMPLES = 128
FEATURE_COUNT = 561


def _lowpass(values: np.ndarray, cutoff_hz: float) -> np.ndarray:
    sos = butter(3, cutoff_hz, btype="lowpass", fs=SAMPLE_RATE, output="sos")
    return sosfiltfilt(sos, values, axis=0)


def _jerk(values: np.ndarray) -> np.ndarray:
    # Preserve the 128-frame contract after differentiating at 50 Hz.
    return np.vstack((np.diff(values, axis=0), np.zeros((1, values.shape[1]), dtype=np.float64))) * SAMPLE_RATE


def _magnitude(values: np.ndarray) -> np.ndarray:
    return np.linalg.norm(values, axis=1, keepdims=True)


def _entropy(values: np.ndarray) -> float:
    bins = min(32, max(2, values.size // 4))
    counts, _ = np.histogram(values, bins=bins)
    probabilities = counts[counts > 0] / values.size
    return float(-np.sum(probabilities * np.log(probabilities)))


def _ar(values: np.ndarray) -> list[float]:
    if np.allclose(values, values[0]):
        return [0.0] * 4
    target = values[4:]
    predictors = np.column_stack([values[4 - lag: -lag] for lag in range(1, 5)])
    return np.linalg.lstsq(predictors, target, rcond=None)[0].astype(float).tolist()


def _time_axis_features(values: np.ndarray) -> list[float]:
    result: list[float] = []
    for fn in (np.mean, np.std, lambda x: np.median(np.abs(x - np.median(x))), np.max, np.min):
        result.extend(float(fn(values[:, axis])) for axis in range(3))
    result.append(float(np.mean(np.sum(np.abs(values), axis=1))))
    for fn in (lambda x: np.mean(x * x), lambda x: np.percentile(x, 75) - np.percentile(x, 25), _entropy):
        result.extend(float(fn(values[:, axis])) for axis in range(3))
    for axis in range(3):
        result.extend(_ar(values[:, axis]))
    result.extend(float(np.corrcoef(values[:, first], values[:, second])[0, 1])
                  if np.std(values[:, first]) > 1e-12 and np.std(values[:, second]) > 1e-12 else 0.0
                  for first, second in ((0, 1), (0, 2), (1, 2)))
    return result


def _time_magnitude_features(values: np.ndarray) -> list[float]:
    signal = values[:, 0]
    return [
        float(np.mean(signal)), float(np.std(signal)),
        float(np.median(np.abs(signal - np.median(signal)))), float(np.max(signal)), float(np.min(signal)),
        float(np.mean(np.abs(signal))), float(np.mean(signal * signal)),
        float(np.percentile(signal, 75) - np.percentile(signal, 25)), _entropy(signal), *_ar(signal),
    ]


def _spectrum(values: np.ndarray) -> np.ndarray:
    # UCI describes 64 FFT bins for each 128-sample window.
    return np.abs(np.fft.fft(values, axis=0)[:64])


def _freq_statistics(signal: np.ndarray) -> list[float]:
    frequencies = np.arange(signal.size, dtype=np.float64) / (signal.size * 2.0)
    weight = np.sum(signal)
    mean_frequency = float(np.dot(frequencies, signal) / weight) if weight > 1e-12 else 0.0
    centered = signal - np.mean(signal)
    scale = float(np.std(signal))
    skew = float(np.mean(centered ** 3) / scale ** 3) if scale > 1e-12 else 0.0
    kurt = float(np.mean(centered ** 4) / scale ** 4 - 3.0) if scale > 1e-12 else 0.0
    return [float(np.mean(signal)), float(np.std(signal)),
            float(np.median(np.abs(signal - np.median(signal)))), float(np.max(signal)), float(np.min(signal)),
            float(np.mean(signal * signal)), float(np.percentile(signal, 75) - np.percentile(signal, 25)),
            _entropy(signal), float(np.argmax(signal) + 1), mean_frequency, skew, kurt]


def _freq_axis_features(values: np.ndarray) -> list[float]:
    spectrum = _spectrum(values)
    result: list[float] = []
    # mean, std, mad, max, min
    for statistic in range(5):
        result.extend(_freq_statistics(spectrum[:, axis])[statistic] for axis in range(3))
    result.append(float(np.mean(np.sum(spectrum, axis=1))))
    # energy, iqr, entropy, max index, mean frequency
    for statistic in range(5, 10):
        result.extend(_freq_statistics(spectrum[:, axis])[statistic] for axis in range(3))
    # UCI's order is skew/kurt paired by axis.
    for axis in range(3):
        stats = _freq_statistics(spectrum[:, axis])
        result.extend(stats[10:12])
    intervals = ((1, 8), (9, 16), (17, 24), (25, 32), (33, 40), (41, 48), (49, 56), (57, 64),
                 (1, 16), (17, 32), (33, 48), (49, 64), (1, 24), (25, 48))
    for axis in range(3):
        for first, last in intervals:
            result.append(float(np.mean(spectrum[first - 1:last, axis] ** 2)))
    return result


def _freq_magnitude_features(values: np.ndarray) -> list[float]:
    signal = _spectrum(values)[:, 0]
    stats = _freq_statistics(signal)
    # Frequency-magnitude groups include SMA between min and energy.
    return [*stats[:5], float(np.mean(np.abs(signal))), *stats[5:]]


def _angle(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator < 1e-12:
        return 0.0
    return float(np.dot(first, second) / denominator)


def extract_uci_features(total_acceleration_g: np.ndarray, gyroscope_rad_s: np.ndarray) -> np.ndarray:
    """Return the ordered 561-value UCI HAR feature contract for one window.

    Inputs are `(128, 3)` arrays in g and radians/second.  A caller with a
    browser `DeviceMotionEvent` should divide acceleration by 9.80665 and
    convert `rotationRate` degrees/second to radians/second first.
    """
    total = np.asarray(total_acceleration_g, dtype=np.float64)
    gyro = np.asarray(gyroscope_rad_s, dtype=np.float64)
    if total.shape != (WINDOW_SAMPLES, 3) or gyro.shape != (WINDOW_SAMPLES, 3):
        raise ValueError("expected exactly 128 samples with 3 acceleration and 3 gyroscope axes")
    if not np.isfinite(total).all() or not np.isfinite(gyro).all():
        raise ValueError("motion samples contain non-finite values")

    gravity = _lowpass(total, 0.3)
    body_acc = total - gravity
    body_acc_jerk = _jerk(body_acc)
    body_gyro_jerk = _jerk(gyro)
    vector: list[float] = []
    for signal in (body_acc, gravity, body_acc_jerk, gyro, body_gyro_jerk):
        vector.extend(_time_axis_features(signal))
    for signal in (body_acc, gravity, body_acc_jerk, gyro, body_gyro_jerk):
        vector.extend(_time_magnitude_features(_magnitude(signal)))
    for signal in (body_acc, body_acc_jerk, gyro):
        vector.extend(_freq_axis_features(signal))
    for signal in (body_acc, body_acc_jerk, gyro, body_gyro_jerk):
        vector.extend(_freq_magnitude_features(_magnitude(signal)))
    gravity_mean = np.mean(gravity, axis=0)
    vector.extend([
        _angle(np.mean(body_acc, axis=0), gravity_mean),
        _angle(np.mean(body_acc_jerk, axis=0), gravity_mean),
        _angle(np.mean(gyro, axis=0), gravity_mean),
        _angle(np.mean(body_gyro_jerk, axis=0), gravity_mean),
        _angle(np.array([1.0, 0.0, 0.0]), gravity_mean),
        _angle(np.array([0.0, 1.0, 0.0]), gravity_mean),
        _angle(np.array([0.0, 0.0, 1.0]), gravity_mean),
    ])
    result = np.asarray(vector, dtype=np.float32)
    if result.shape != (FEATURE_COUNT,):
        raise RuntimeError(f"feature builder created {result.size} values; expected {FEATURE_COUNT}")
    return result


def fit_calibration(raw_features: np.ndarray, model_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Legacy per-feature affine fit, retained for comparing adapter quality."""
    if raw_features.shape != model_features.shape or raw_features.shape[1] != FEATURE_COUNT:
        raise ValueError("calibration matrices must both be N x 561")
    raw_mean = raw_features.mean(axis=0)
    target_mean = model_features.mean(axis=0)
    centered = raw_features - raw_mean
    denominator = np.sum(centered * centered, axis=0)
    gain = np.divide(np.sum(centered * (model_features - target_mean), axis=0), denominator,
                     out=np.zeros(FEATURE_COUNT, dtype=np.float64), where=denominator > 1e-12)
    offset = target_mean - gain * raw_mean
    return gain.astype(np.float32), offset.astype(np.float32)


def apply_calibration(features: np.ndarray, gain: np.ndarray, offset: np.ndarray) -> np.ndarray:
    if features.shape != (FEATURE_COUNT,) or gain.shape != (FEATURE_COUNT,) or offset.shape != (FEATURE_COUNT,):
        raise ValueError("feature calibration must have 561 values")
    return (features * gain + offset).astype(np.float32)


def fit_linear_adapter(raw_features: np.ndarray, model_features: np.ndarray,
                       ridge: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Fit a regularized full-feature bridge from raw signals to UCI's contract.

    The public UCI feature matrix was generated by a proprietary-era MATLAB
    pipeline; a one-column-at-a-time scale/offset fit loses correlations between
    signal families.  This ridge adapter uses those correlations, while staying
    linear so it can be fused into the first MLP Gemm for zero extra runtime
    graph steps.
    """
    if raw_features.shape != model_features.shape or raw_features.shape[1] != FEATURE_COUNT:
        raise ValueError("adapter matrices must both be N x 561")
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    raw_mean = raw_features.mean(axis=0, dtype=np.float64)
    target_mean = model_features.mean(axis=0, dtype=np.float64)
    centered_raw = raw_features.astype(np.float64) - raw_mean
    centered_target = model_features.astype(np.float64) - target_mean
    gram = centered_raw.T @ centered_raw
    mapping = np.linalg.solve(gram + ridge * np.eye(FEATURE_COUNT),
                              centered_raw.T @ centered_target)
    offset = target_mean - raw_mean @ mapping
    return mapping.astype(np.float32), offset.astype(np.float32)


def apply_linear_adapter(features: np.ndarray, mapping: np.ndarray, offset: np.ndarray) -> np.ndarray:
    if (features.shape != (FEATURE_COUNT,) or mapping.shape != (FEATURE_COUNT, FEATURE_COUNT)
            or offset.shape != (FEATURE_COUNT,)):
        raise ValueError("linear feature adapter must map 561 values to 561 values")
    return (features @ mapping + offset).astype(np.float32)


def load_raw_window(dataset: Path, partition: str, index: int) -> tuple[np.ndarray, np.ndarray]:
    """Load a raw UCI window for calibration/tests; not used during live capture."""
    signals = dataset / partition / "Inertial Signals"
    suffix = "train" if partition == "train" else "test"
    def read(prefix: str) -> np.ndarray:
        return np.loadtxt(signals / f"{prefix}_{suffix}.txt", max_rows=1, skiprows=index)
    total = np.column_stack([read(f"total_acc_{axis}") for axis in "xyz"])
    gyro = np.column_stack([read(f"body_gyro_{axis}") for axis in "xyz"])
    return total, gyro
