# StrataMotion metric review

## The meaningful weakness

The engine's microsecond forward-pass latency was already strong for this
fixed-shape MLP. The product weakness was **model quality without a safe
decision policy**: the original demo model reached 91.69% accuracy on UCI's
provided held-out test split, so roughly one in twelve activity labels was
wrong while the UI always displayed a definitive answer.

## Improvement shipped

The model is now trained on feature-standardized data. Instead of adding a
runtime preprocessing dependency, the normalization is algebraically folded
into the first `Gemm` weight and bias before ONNX export. The C++ engine still
runs the same supported `Gemm -> Relu -> Gemm` graph, with no new operation,
allocation, or Python dependency on its inference path.

| Metric | Original model | Current model | Change |
|---|---:|---:|---:|
| UCI held-out activity accuracy | 91.69% | **94.84%** | **+3.15 points** |
| Held-out error rate | 8.31% | **5.16%** | **37.9% lower** |
| Weakest-class (SITTING) recall | 89.21% | **91.24%** | **+2.03 points** |
| Engine model contract | 561 → 64 → 6 | 561 → 64 → 6 | unchanged |
| Custom-engine model oracle | pass | **pass** | unchanged correctness gate |

## Confidence gate

The dashboard now exposes a practical policy: **Accept** when the model's
softmax confidence meets the measured threshold; otherwise return **Review**
instead of presenting a low-confidence label as trustworthy. On the current
held-out evaluation, the threshold is 0.64: it accepts 92.6% of examples with
97.07% selective accuracy. This is useful because the intended live sensor
scenario has phone placement and orientation shift; a cautious abstention is
better than a confident-looking incorrect activity label.

The threshold and selective metrics are chosen/reported on the same held-out
split, so they are a transparent demo quality signal—not an independent
production guarantee. A release should choose that threshold on a separate
validation set and keep a final untouched test set.

## Input-distribution drift guard

Confidence alone cannot detect every bad sensor window. A model can be
confident even when a phone is mounted differently, has a unit-conversion bug,
or produces values unlike its training data. The runtime now compares all 561
input values with a robust 1st–99th percentile envelope fitted on UCI training
data. It reports **In range** or **Shifted** in the dashboard, and either low
confidence *or* shifted input changes the decision to **Review**.

The feature-outlier budget is 17.40%, chosen from the fused adapter's known
raw-window output distribution. It retains 99.2% of the 256 disjoint raw-UCI
validation windows. On a separate 60-window sensor-guard check, it reviewed
only 1.7% of normal windows and caught 90.0% on average across three simulated
unit/scale corruptions (accelerometer x10, gyroscope x20, and both). This is
the right comparison after fixing the adapter/guard mismatch: before any
detector, corruption detection was 0.0%.

## Live-feature fidelity

The original per-column gain/offset bridge was only assessed on its fitting
windows. The replacement is a ridge adapter trained on 512 raw UCI train
windows and fused algebraically into the live model's first `Gemm`, so the
C++ runtime still executes only `Gemm -> Relu -> Gemm`. On 256 disjoint raw
UCI test windows it reduces reconstruction MAE from **about 0.10** (the old
adapter's unseen-window result) to **0.0679**: a **32% reduction**. It is a
real improvement, but it has **not** reached the `<0.02` target.

The fused live graph gets 90.62% accuracy on that raw-UCI proxy subset. That
is useful integration evidence, not a phone accuracy claim: **real phone
accuracy has not been verified** because there are no labelled recordings from
the target phone, placement, or browser.

## INT8 deployment trade-off

`strata_int8_infer` now performs real dynamic-activation INT8 inference using
the engine's signed-int8 matrix-vector kernel. It quantizes weights once at
load time, retains FP32 biases, and reports its packed deployment footprint.
On the full UCI held-out split it achieves **94.71%** (2,791/2,947), within
0.14 percentage points of the FP32 model's 94.84%, while reducing the deployed
weight/bias/scales footprint from the 142.5 KiB FP32 ONNX artifact to **35.7
KiB (74.9% smaller)**.

The measured latency trade-off is important: on the same 2,000-run local
measurement, FP32 was 2.56 us mean / 3.30 us P99 and INT8 was 3.58 us mean /
4.50 us P99. Dynamic activation quantization adds work on this AVX2 CPU, so
this is a memory-constrained deployment option—not a dishonest speed claim.
