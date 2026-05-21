#!/usr/bin/env python3
# =============================================================================
#  Generates the Phase 3 test asset:
#    models/mlp.onnx     — a 2-layer MLP  (Gemm -> Relu -> Gemm)
#    models/mlp_io.csv   — a fixed input row + the numpy-computed expected
#                          output row (the correctness oracle for the C++ side)
#
#  Weights are deterministic (fixed seed) so the .onnx and the oracle are
#  reproducible and the C++ test needs no Python at build/test time.
# =============================================================================
import os
import numpy as np
from onnx import helper, TensorProto, numpy_helper, checker, save

K, H, O = 16, 32, 8          # input / hidden / output widths
rng = np.random.default_rng(42)

W1 = rng.standard_normal((H, K)).astype(np.float32) * 0.25
B1 = rng.standard_normal((H,)).astype(np.float32) * 0.10
W2 = rng.standard_normal((O, H)).astype(np.float32) * 0.25
B2 = rng.standard_normal((O,)).astype(np.float32) * 0.10

def init(name, arr):
    return numpy_helper.from_array(arr, name)

inp = helper.make_tensor_value_info("input",  TensorProto.FLOAT, [1, K])
out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, O])

nodes = [
    helper.make_node("Gemm", ["input", "W1", "B1"], ["h0"], transB=1),
    helper.make_node("Relu", ["h0"], ["h1"]),
    helper.make_node("Gemm", ["h1", "W2", "B2"], ["output"], transB=1),
]

graph = helper.make_graph(
    nodes, "strata_mlp", [inp], [out],
    initializer=[init("W1", W1), init("B1", B1),
                 init("W2", W2), init("B2", B2)])

model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 9          # ORT 1.17 supports IR <= 9
checker.check_model(model)

here   = os.path.dirname(os.path.abspath(__file__))
models = os.path.join(here, "..", "models")
os.makedirs(models, exist_ok=True)
save(model, os.path.join(models, "mlp.onnx"))

# numpy oracle for a fixed, reproducible input.
x = (rng.standard_normal((1, K)).astype(np.float32))
h = np.maximum(x @ W1.T + B1, 0.0)
y = h @ W2.T + B2

with open(os.path.join(models, "mlp_io.csv"), "w", encoding="utf-8") as f:
    f.write(",".join(f"{v:.8e}" for v in x.ravel()) + "\n")
    f.write(",".join(f"{v:.8e}" for v in y.ravel()) + "\n")

print(f"wrote models/mlp.onnx  (K={K} H={H} O={O})")
print(f"wrote models/mlp_io.csv  in={x.size} out={y.size}")
print("expected output:", np.array2string(y.ravel(), precision=5))
