# StrataMotion: local edge-inference demo

## What it demonstrates

StrataMotion turns the engine into something a person can interact with. It
replays 72 held-out sensor-feature windows, one at a time, through a persistent
`strata_stream` C++ process. The local dashboard displays the predicted
activity, confidence, selected motion-feature values, C++ wall-clock latency,
and the diagnostic timing for the compiled `Gemm -> Relu -> Gemm` plan.

There is no browser-side ML and no Python inference path. Python only downloads
the dataset, trains/exports the model, and hosts a local-only HTTP bridge. The
engine process loads the ONNX weights once at start-up and receives a CSV window
over stdin for each dashboard update.

The dashboard labels its timing **Instrumented latency** because it includes
the diagnostic per-step trace. Use `strata_infer --profile` for clean hot-path
latency percentiles; tracing is intentionally outside that measurement path.

## Live-device path

The **Use phone sensors** control is a second, real input path—not a cosmetic
animation. After the user grants the browser's normal motion permission, the
page takes 128 `DeviceMotionEvent` samples (2.56 seconds at 50 Hz), converts
acceleration from m/s² to g and gyro speed from degrees/s to radians/s, and
posts those values to the local server. `demo/har_features.py` derives the
documented UCI HAR body/gravity, jerk, magnitude, FFT, band-energy, and angle
features in UCI's **ordered 561-value feature contract**. The server then
uses a reproducible affine adapter fitted against UCI's raw training windows
before streaming the vector into `strata_stream`.

This is intentionally honest about what is and is not validated:

- The local end-to-end path is verified with a raw UCI inertial window: 128
  samples → 561 values → persistent C++ runner → JSON prediction.
- The feature adapter is fitted using 96 public UCI training windows. Its
  in-sample mean absolute reconstruction error is recorded in
  `data/demo/metadata.json`; it is a compatibility adapter, not a claim that
  arbitrary phones will reach the replay test accuracy.
- Device orientation, browser sensor units, phone placement, and distribution
  shift can change the prediction. A production mobile release needs on-device
  calibration and evaluation on new labelled recordings.
- The server is deliberately bound to `127.0.0.1`. A phone on a different
  device cannot reach a laptop's loopback address; a production LAN/mobile
  bridge requires authenticated HTTPS and is not claimed by this demo.

## Run it

From the repository root on Windows:

```bat
scripts\run-stratamotion.bat
```

Then open `http://127.0.0.1:8080`. The server binds only to `127.0.0.1`; it is
not exposed to the network. Press `Ctrl+C` in that terminal when the demo ends.

The first run downloads and safely extracts the public data, trains the
deterministic seeded model, builds the C++ tools, and launches the dashboard.
Later runs reuse the dataset but retrain the model so the model artifact is
always reproducible from the tracked code.

## Dataset, permission, and attribution

This demo uses **Human Activity Recognition Using Smartphones**, UCI Machine
Learning Repository dataset 240, by Reyes-Ortiz, Anguita, Ghio, Oneto, and
Parra (2013), DOI: [10.24432/C54S4K](https://doi.org/10.24432/C54S4K).

UCI marks this dataset as [Creative Commons Attribution 4.0 International
(CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). That license allows
sharing and adapting the data for personal, educational, and commercial work,
as long as you provide appropriate credit, link the license, and indicate
changes. This project does that by retaining this attribution and the source
link in the dashboard metadata. Keep this attribution if you publish a fork or
a screen recording using the dataset.

The official source has 10,299 labelled motion windows from 30 people carrying
a waist-mounted Samsung Galaxy S II. It contains preprocessed 561-feature
vectors calculated from tri-axial accelerometer and gyroscope signals; it is
not live hardware data and does not identify a person in this demo.

## Honest boundaries

- This is an activity classifier for six labelled activities, not a medical,
  fall-detection, workplace-safety, or surveillance product.
- The reported test accuracy is the model’s result on UCI's provided held-out
  split. It is not a claim about a new user, a different phone, or a production
  environment.
- The project now includes a local browser sensor-ingestion/preprocessing path.
  It is not a production mobile deployment: a phone on another device needs an
  authenticated HTTPS bridge and real-device calibration before it is credible.
- The data and generated model are excluded from Git because the scripts
  regenerate them. `models/mlp.onnx` remains the small committed engine test
  model.

## Demo flow for a recruiter

1. Run `scripts\run-stratamotion.bat` and open the shown local URL.
2. Click **Play replay**. Each 900 ms update is a separate real held-out data
   window flowing to the already-loaded C++ engine.
3. Point out the ground-truth check, latency, class-score bars, and execution
   trace. A mismatch is shown honestly rather than hidden.
4. In the terminal, stop the dashboard with `Ctrl+C`, then run the direct
   `strata_infer` command from the README to show the oracle, repeatability,
   and scalar/AVX2-equivalence checks.
