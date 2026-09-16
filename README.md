# StrataCompute

**A verifiable C++20 inference engine for small, latency-sensitive MLPs.**

StrataCompute loads a deliberately small ONNX subset and runs it without
Python or ONNX Runtime on the inference path. It preloads weights, compiles a
flat execution plan, reuses a preallocated activation arena, and executes
hand-written AVX2/FMA matrix-vector kernels. The result is a focused systems
project for fixed-shape, single-sample `Gemm -> Relu -> Gemm` models - not a
general ONNX runtime or web service.

## Why it exists

Small control, sensor, and embedded-style ML workloads can spend more time in
runtime overhead than in arithmetic. StrataCompute moves model parsing, weight
packing, plan construction, and memory allocation out of the hot path. Each
inference is then matrix-vector work plus elementwise operations on already
allocated memory.

The project includes a CLI, deterministic repeat checks, NumPy oracle checks,
scalar-versus-SIMD equivalence checks, operation tracing, CTest coverage, and
a repeatable benchmark path.

## What is supported

- Static-shape FP32 input/output and a sequential `Gemm` (`transB=1`, optional
  bias) / `Relu` graph.
- AVX2/FMA FP32 kernels plus a scalar reference path.
- A dynamic-activation INT8 execution mode with packed INT8 weights.
- Explicit failure for unsupported graphs such as convolution, attention,
  branching, dynamic shapes, and arbitrary quantized ONNX graphs.

## StrataMotion demo

`StrataMotion` turns the engine into a local machine-health/activity-monitor
demo. It replays held-out UCI Human Activity Recognition windows in a browser
dashboard and sends every prediction to the persistent custom C++ runner.
Replay works without hardware. The optional phone path collects one 2.56-second
motion window in a compatible browser and keeps the request local.

The raw-sensor adapter is fused into the first model layer, so live C++
inference still uses only `Gemm -> Relu -> Gemm`. The local Python server still
constructs the 561 sensor features, but no longer applies an additional adapter
matrix per request. UCI HAR is CC BY 4.0; see
[the demo guide](docs/stratamotion-demo.md) for attribution and safe-demo
boundaries.

```bat
scripts\run-stratamotion.bat
```

Then open http://127.0.0.1:8080.

## Measured results

All model-quality figures use UCI HAR's provided held-out split unless stated
otherwise. Re-run the commands to reproduce them; microsecond timing varies by
CPU, compiler, power state, and OS scheduling.

| Result | Before | Current | Evidence |
|---|---:|---:|---|
| Activity accuracy | 91.69% | **94.84%** | Held-out UCI test split |
| Error rate | 8.31% | **5.16%** | 37.9% lower |
| Weakest class: SITTING recall | 89.21% | **91.24%** | Class-balanced training candidate |
| Raw-to-feature MAE | about 0.10 | **0.0679** | 256 unseen raw-UCI windows |
| Packed model footprint | 142.5 KiB FP32 | **35.7 KiB INT8** | 74.9% smaller |
| INT8 accuracy | - | **94.71%** | 2,791 / 2,947 held-out samples |

The live raw-UCI proxy path scores 90.62% on its 256-window subset. This is
integration evidence, not a phone claim: real-phone accuracy is **not yet
verified** because labelled recordings from the target device, placement, and
browser have not been collected. Likewise, the adapter has improved but has
not met the `<0.02` MAE target.

INT8 is a memory trade-off on the current AVX2 machine, not a speed claim. In
the same 2,000-run local comparison, FP32 measured 2.56 us mean / 3.30 us P99,
while dynamic INT8 measured 3.58 us / 4.50 us. See
[the full metrics review](docs/metrics.md) for methodology and guard results.

## Run it

Build and run all tests on Windows with Visual Studio 2019 Build Tools:

```bat
scripts\build-msvc.bat Release
```

Run the committed oracle model through the custom engine:

```bat
build\tools\strata_infer.exe models\mlp.onnx models\mlp_io.csv ^
  --oracle --verify-repeats 1000 --iterations 1000 --profile ^
  --compare-backends --trace
```

After running the StrataMotion launcher, validate the trained activity model:

```bat
build\tools\strata_infer.exe models\har_activity.onnx models\har_activity_io.csv ^
  --oracle --verify-repeats 100 --iterations 500 --profile --compare-backends --trace
```

Measure the actual INT8 execution path, footprint, output delta, and held-out
accuracy:

```bat
build\tools\strata_int8_infer.exe models\har_activity.onnx ^
  "data\uci_har\UCI HAR Dataset\test\X_test.txt" ^
  --iterations 2000 --verify-fp32 ^
  --labels "data\uci_har\UCI HAR Dataset\test\y_test.txt"
```

For the optional ONNX Runtime/PyTorch comparison, run
`scripts\run-phase5.bat`. Docker and Kubernetes benchmark templates remain in
the repository; they are batch benchmark tooling, not a deployed web product.

## Project map

| Location | Purpose |
|---|---|
| `include/strata`, `src/` | Memory pool, ONNX reader, SIMD kernels, FP32 and INT8 engines |
| `tools/` | `strata_infer`, `strata_int8_infer`, and persistent `strata_stream` CLIs |
| `demo/` | Local StrataMotion dashboard and raw-sensor feature bridge |
| `scripts/` | Build, dataset, training, evaluation, and benchmark commands |
| `tests/` | Unit and end-to-end engine checks |
| `docs/` | Benchmark, metric, and demo methodology |

The key architecture rule is simple: model load and memory setup happen once;
the custom forward path performs no heap allocations.
