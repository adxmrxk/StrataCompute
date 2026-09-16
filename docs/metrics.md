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
| UCI held-out activity accuracy | 91.69% | **94.67%** | **+2.98 points** |
| Held-out error rate | 8.31% | **5.33%** | **35.9% lower** |
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

The feature-outlier budget is 17.11%, chosen as the 99th percentile of known
training-window outlier rates. It retains 99.8% of the provided held-out UCI
test windows, while making obvious distribution shifts visible rather than
silently feeding them to the model as though they were normal data.

## Remaining measured boundary

The live feature adapter reconstructs the UCI training representation with
0.0739 mean absolute error on its 96 fitting windows. The raw-window → 561
feature → C++ inference path is exercised locally, but **live-phone accuracy
has not been verified**. New labelled phone recordings are the next necessary
metric before claiming a deployed activity product.
