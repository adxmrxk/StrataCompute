"""Local no-hardware dashboard server for the StrataMotion demo.

Run after building the C++ tools and training the demo model:
    python demo/server.py
Then browse http://127.0.0.1:8080 .  The server keeps strata_stream alive,
so every replay window is evaluated by the actual custom C++ engine.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import threading
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from har_features import apply_linear_adapter, extract_uci_features

ROOT = Path(__file__).resolve().parents[1]
DEMO_DATA = ROOT / "data" / "demo"
MODEL = ROOT / "models" / "har_activity.onnx"
LIVE_MODEL = ROOT / "models" / "har_activity_live.onnx"
LABELS = ROOT / "demo" / "activity_labels.txt"


class ReplayEngine:
    def __init__(self) -> None:
        required = [DEMO_DATA / "activity_replay.csv", DEMO_DATA / "activity_replay_labels.csv",
                    DEMO_DATA / "metadata.json", DEMO_DATA / "feature_adapter.npz",
                    DEMO_DATA / "input_health.npz", MODEL, LIVE_MODEL, LABELS]
        missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
        if missing:
            raise RuntimeError("missing demo assets: " + ", ".join(missing) +
                               ". Run python scripts/get_uci_har.py then python scripts/train_har_demo.py")
        self.rows = list(csv.reader((DEMO_DATA / "activity_replay.csv").open(encoding="utf-8", newline="")))
        self.truth = [int(line.strip()) for line in (DEMO_DATA / "activity_replay_labels.csv").read_text().splitlines()]
        self.labels = [line.strip() for line in LABELS.read_text().splitlines() if line.strip()]
        self.metadata = json.loads((DEMO_DATA / "metadata.json").read_text(encoding="utf-8"))
        adapter = np.load(DEMO_DATA / "feature_adapter.npz")
        self.mapping = adapter["mapping"].astype(np.float32)
        self.offset = adapter["offset"].astype(np.float32)
        health = np.load(DEMO_DATA / "input_health.npz")
        self.lower = health["lower"].astype(np.float32)
        self.upper = health["upper"].astype(np.float32)
        self.outlier_threshold = float(health["threshold"])
        if len(self.rows) != len(self.truth):
            raise RuntimeError("replay feature/label files have different lengths")
        binary = next((path for path in [ROOT / "build" / "tools" / "strata_stream.exe",
                                         ROOT / "build-onnx" / "tools" / "strata_stream.exe"] if path.exists()), None)
        if binary is None:
            raise RuntimeError("strata_stream.exe is missing. Run scripts\\build-msvc.bat Release")
        self.process = self._start(binary, MODEL)
        # The adapter is fused into the first Gemm: live capture still has the
        # same three engine steps as replay, without Python applying a 561x561
        # matrix for every request before C++ inference.
        self.live_process = self._start(binary, LIVE_MODEL)
        self.position = 0
        self.lock = threading.Lock()

    def _start(self, binary: Path, model: Path) -> subprocess.Popen[str]:
        return subprocess.Popen(
            [str(binary), str(model), "--labels", str(LABELS), "--trace"],
            cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1,
        )

    @staticmethod
    def _close(process: subprocess.Popen[str]) -> None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()

    def close(self) -> None:
        self._close(self.process)
        self._close(self.live_process)

    def status(self) -> dict:
        return {"position": self.position, "total": len(self.rows), "metadata": self.metadata}

    def reset(self) -> dict:
        with self.lock:
            self.position = 0
            return self.status()

    def next(self) -> dict:
        with self.lock:
            index = self.position
            self.position = (self.position + 1) % len(self.rows)
            features = np.asarray(self.rows[index], dtype=np.float32)
            result = self._infer(features)
            actual_index = self.truth[index]
            visual = self.metadata["visual_features"]
            result.update({
                "sample": index + 1,
                "total": len(self.rows),
                "ground_truth_index": actual_index,
                "ground_truth": self.labels[actual_index],
                "correct": result["prediction_index"] == actual_index,
                "visual_features": [
                    {"name": item["name"], "value": float(features[item["index"]])}
                    for item in visual
                ],
                "live": False,
            })
            return result

    def _infer(self, features: np.ndarray, sensor_sanity: str = "not sampled",
               *, process: subprocess.Popen[str] | None = None,
               guard_features: np.ndarray | None = None) -> dict:
        runner = self.process if process is None else process
        if runner.poll() is not None:
            detail = runner.stderr.read().strip()
            raise RuntimeError("C++ inference runner exited" + (f": {detail}" if detail else ""))
        if features.shape != (561,):
            raise RuntimeError(f"feature extractor returned {features.size} values; expected 561")
        if guard_features is None:
            guard_features = features
        assert runner.stdin is not None and runner.stdout is not None
        runner.stdin.write(",".join(f"{value:.9g}" for value in features) + "\n")
        runner.stdin.flush()
        response = runner.stdout.readline()
        if not response:
            raise RuntimeError("C++ inference runner returned no response")
        result = json.loads(response)
        if not result.get("ok"):
            raise RuntimeError(result.get("error", "C++ inference failed"))
        minimum = float(self.metadata["quality_guard"]["minimum_confidence"])
        outlier_fraction = float(np.mean((guard_features < self.lower) | (guard_features > self.upper)))
        distribution_ok = outlier_fraction <= self.outlier_threshold
        reason = []
        if result["confidence"] < minimum:
            reason.append("low confidence")
        if not distribution_ok:
            reason.append("input shifted")
        if sensor_sanity == "invalid":
            reason.append("sensor magnitude implausible")
        result["decision"] = "accepted" if not reason else "review"
        result["minimum_confidence"] = minimum
        result["outlier_fraction"] = outlier_fraction
        result["maximum_outlier_fraction"] = self.outlier_threshold
        result["input_fit"] = "in range" if distribution_ok else "shifted"
        result["sensor_sanity"] = sensor_sanity
        result["review_reason"] = ", ".join(reason)
        return result

    def live(self, samples: object) -> dict:
        if not isinstance(samples, list) or len(samples) != 128:
            raise RuntimeError("live capture needs exactly 128 motion samples (2.56 seconds at 50 Hz)")
        try:
            acceleration = np.asarray([sample["acceleration_g"] for sample in samples], dtype=np.float64)
            gyroscope = np.asarray([sample["gyroscope_rad_s"] for sample in samples], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("each live sample needs three acceleration_g and gyroscope_rad_s values") from error
        max_acceleration_g = float(np.linalg.norm(acceleration, axis=1).max())
        max_gyroscope_rad_s = float(np.linalg.norm(gyroscope, axis=1).max())
        # Conservative bounds sit above every UCI held-out raw window. They
        # catch the practical browser-integration failures: treating m/s² as g
        # or degrees/s as radians/s before those bad values reach the model.
        sensor_sanity = "in range" if max_acceleration_g <= 4.0 and max_gyroscope_rad_s <= 6.0 else "invalid"
        raw_features = extract_uci_features(acceleration, gyroscope)
        adapted_features = apply_linear_adapter(raw_features, self.mapping, self.offset)
        with self.lock:
            result = self._infer(raw_features, sensor_sanity, process=self.live_process,
                                 guard_features=adapted_features)
        visual = self.metadata["visual_features"]
        result.update({
            "live": True,
            "source": "live device-motion capture",
            "maximum_acceleration_g": max_acceleration_g,
            "maximum_gyroscope_rad_s": max_gyroscope_rad_s,
            "visual_features": [{"name": item["name"], "value": float(adapted_features[item["index"]])}
                                for item in visual],
        })
        return result


ENGINE: ReplayEngine | None = None


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, directory=str(ROOT / "demo"), **kwargs)

    def log_message(self, fmt: str, *args) -> None:
        print("dashboard:", fmt % args)

    def send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 (HTTP method name)
        global ENGINE
        route = urlparse(self.path).path
        try:
            if route == "/api/status":
                assert ENGINE is not None
                self.send_json(ENGINE.status())
            elif route == "/api/next":
                assert ENGINE is not None
                self.send_json(ENGINE.next())
            elif route == "/":
                self.path = "/index.html"
                super().do_GET()
            else:
                super().do_GET()
        except Exception as error:
            self.send_json({"ok": False, "error": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:  # noqa: N802
        global ENGINE
        try:
            assert ENGINE is not None
            route = urlparse(self.path).path
            if route == "/api/reset":
                self.send_json(ENGINE.reset())
            elif route == "/api/live":
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length <= 0 or content_length > 250_000:
                    raise RuntimeError("live capture payload is missing or too large")
                payload = json.loads(self.rfile.read(content_length))
                self.send_json(ENGINE.live(payload.get("samples")))
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as error:
            self.send_json({"ok": False, "error": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)


def main() -> int:
    global ENGINE
    ENGINE = ReplayEngine()
    server = ThreadingHTTPServer(("127.0.0.1", 8080), Handler)
    print("StrataMotion dashboard: http://127.0.0.1:8080")
    print("Ctrl+C stops the local server and the C++ inference runner.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard.")
    finally:
        server.server_close()
        ENGINE.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"dashboard: {error}", file=sys.stderr)
        raise SystemExit(1)
