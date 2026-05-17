# StrataCompute

Bare-metal C++20 ML inference engine — deterministic, sub-millisecond tail
latency by bypassing the Python runtime, owning memory, and hand-rolling
SIMD linear algebra.

## Status

| Phase | Scope | State |
|-------|-------|-------|
| **1** | `Strata::MemoryPool`, `Strata::Tensor`, build system, tests | ✅ done — zero warnings under `/W4 /WX` |
| **2** | Hand-rolled SIMD math kernels (AVX2/FMA + scalar ref), CPUID dispatch, Google Benchmark | ✅ done — 19/19 tests pass, 6–9× speed-up measured |
| **3** | ONNX Runtime C++ API baseline: model loader, I/O introspection, graph runner | ✅ done — 3/3 tests pass vs numpy oracle |
| **4** | Custom forward-pass: ONNX weight extraction → MemoryPool/Tensor, pure-C++ Gemm/Relu graph loop on Strata::Compute kernels | ✅ done — 23/23 offline tests pass |
| **5** | Microsecond benchmarking: PyTorch vs ONNX RT (C++) vs StrataCompute, mean/P95/P99 + memory model | ✅ done — see [docs/benchmarks.md](docs/benchmarks.md) |

**Headline (single-sample MLP, single-thread, identical weights/input):**

| Engine | mean | p99 |
|---|--:|--:|
| PyTorch (Python) | 20.97 µs | 84.90 µs |
| ONNX Runtime (C++) | 6.99 µs | 19.90 µs |
| **StrataCompute (C++)** | **0.12 µs** | **0.20 µs** |

~172× lower mean and ~425× lower P99 tail vs PyTorch; ~57× vs ONNX Runtime —
with output verified bit-close to the numpy oracle (max abs err 1.8e-7). **All
five phases complete.**

### Phase 2 measured acceleration (i5-12400, MSVC `/O2 /arch:AVX2`)

Hand-rolled AVX2/FMA vs the auto-vectorisation-suppressed scalar reference
(`bench_matvec`, matched `{M,K}`):

| Kernel | 256² | 512² | 1024² |
|---|---|---|---|
| FP32 scalar → AVX2 | 6.7× | 6.7× | 5.9× |
| INT8 scalar → AVX2 | 7.0× | 8.4× | 9.0× |

INT8 AVX2 sustains ~28 G MAC/s; FP32 AVX2 ~12–19 G/s.

## Toolchain — deviations from the original spec

The spec assumed a Linux/GCC, CMake, AVX-512 environment. The actual dev box
required documented, user-approved substitutions (functionally equivalent):

| Spec | This repo | Why |
|------|-----------|-----|
| GCC `-O3 -march=native -flto -Wall -Wextra -Werror`, C++20 | **MSVC 19.29 (VS2019 Build Tools)** `/O2 /GL /W4 /WX /std:c++20 /arch:AVX2` | Only compiler present was MinGW GCC 6.3 (2016, 32-bit, no C++20). VS2019 Build Tools has full C++20. |
| CMake | **CMake via `pip install cmake`** (4.3.2) | No CMake was installed; no admin needed for the pip route. |
| AVX-512 ("64 INT8 ops/clock") | **AVX2 + FMA + VNNI**, with a runtime-dispatch seam for a future AVX-512 kernel | Dev CPU is an i5-12400 (Alder Lake) — Intel fused AVX-512 *off* on 12th-gen consumer parts. |
| Google Benchmark linked always | Fetched via `FetchContent`, **`STRATA_BUILD_BENCHMARKS=OFF` by default** | Keeps Phase 1 buildable offline; benchmarks (Phase 2) need network at configure time. |

Compiler flags are abstracted in `CMakeLists.txt` (`strata_flags` INTERFACE
target): the **GCC/Clang `-O3 -march=native -flto` path is retained** so the
same tree builds on AVX-512 deployment hardware unchanged.

## Docker (recommended — realises the spec's original toolchain)

The Windows host forced the MSVC/AVX2 substitution. The Linux container
instead builds via the **non-MSVC CMake branch**, i.e. the spec's *original*
intended toolchain — GCC with `-O3 -march=native -flto -Wall -Wextra
-Wpedantic -Werror`, C++20 — and bundles **Valgrind**, so the real L1/LL
cache-miss counts the spec asked for (impossible to capture on Windows) can
finally be measured.

```sh
docker build -t stratacompute .                 # builds + runs P1/2/4 (ctest gate)
docker run --rm stratacompute                   # test + phase5 + cachegrind
docker run --rm stratacompute test              # just the suites
docker run --rm -v "$PWD/out:/app/out" stratacompute phase5      # collect reports
docker run --rm -v "$PWD/out:/app/out" stratacompute cachegrind  # real cache misses
```

The offline suite (Phases 1, 2, 4) is compiled **and run** during
`docker build`, so the image fails to build if anything regresses under
GCC `-Werror` — the build is its own verification gate. (`-march=native`
targets the build host's CPU; run the image on that host.)

> Status: the Docker artifacts ([Dockerfile](Dockerfile),
> [scripts/docker-entrypoint.sh](scripts/docker-entrypoint.sh),
> [.dockerignore](.dockerignore)) and the cross-platform ONNX CMake fix are
> complete and reviewed, but the image was **not built/run here** — the
> Docker daemon is not running in this environment. The GCC `-Werror` build
> is therefore exercised by `docker build`, not pre-verified on this box.

## Kubernetes

StrataCompute is a **batch workload** — it runs the suite + Phase 5 benchmark
+ Cachegrind to completion and exits. The correct K8s primitives are therefore
a **`Job`** (one-shot run) and a **`CronJob`** (scheduled latency-drift watch),
**not** a `Deployment`/`Service` — there is no server, no port, nothing to
load-balance; a Deployment would just crash-loop on the process exiting. The
manifests in [k8s/](k8s/) reflect that deliberately.

```sh
# Prereq: image built (Docker) and pushed to a registry the cluster can pull,
# or loaded into kind/minikube. Point the kustomization at your image by
# editing the `images:` block in k8s/kustomization.yaml (newName/newTag).

kubectl apply -k k8s/
kubectl -n stratacompute logs job/stratacompute-bench -f
kubectl -n stratacompute cp <pod>:/app/out ./out     # pull reports
kubectl delete -k k8s/
```

What the manifests encode (this is an HFT-latency project, so these matter):

- **Guaranteed QoS** — `requests == limits`, **integer CPU** — so the pod is
  eligible for the kubelet **static CPU Manager**. Sub-microsecond P99 is only
  trustworthy with `--cpu-manager-policy=static` + CFS-throttling off +
  NUMA-local isolated cores on the node. In a default shared pool the mean is
  indicative but the **tail will be noisy** — report it as such.
- **`-march=native` portability** — the image is compiled for the build
  host's CPU; pin nodes via `nodeSelector`/affinity to a matching CPU, or
  rebuild with a fixed `-march` for heterogeneous clusters (else SIGILL).
- `restartPolicy: Never` + `backoffLimit: 1` so a failed correctness gate
  surfaces instead of silently retrying; PVC keeps reports after pod exit.

> Status: manifests + kustomization are written and pass **offline render
> validation** (`kubectl kustomize k8s/` → all four resources, exit 0). They
> were **not applied to a cluster** — there is no cluster here (and the Docker
> daemon is down, so the image isn't built/pushed yet). Field-level schema
> validation needs a live API server and is therefore not done on this box.

## Build (Windows / MSVC)

VS2019 Build Tools is the *Build Tools* SKU (no IDE), which CMake's
`Visual Studio` generator can't auto-discover via `vswhere`. The build is
therefore driven through `vcvars64` + the **NMake Makefiles** generator.

```bat
scripts\build-msvc.bat Release
```

This configures `build/`, compiles, and runs the test suite. Manual equivalent
(from a `vcvars64`-initialised shell, with the pip CMake on `PATH`):

```
cmake -S . -B build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

## Layout

```
include/strata/         memory_pool.hpp, tensor.hpp
include/strata/compute/  matvec.hpp, cpu_features.hpp
include/strata/onnx/     onnx_model.hpp
src/                     memory_pool.cpp
src/compute/             cpu_features, matvec_{scalar,avx2,dispatch}.cpp
include/strata/engine/   forward_engine.hpp
src/onnx/                onnx_model.cpp (ORT pImpl), onnx_graph.cpp (reader)
src/engine/              forward_engine.cpp (custom pure-C++ forward pass)
cmake/                   onnxruntime.cmake (prebuilt ORT fetch)
models/                  mlp.onnx + mlp_io.csv (generated, committed)
tests/                   offline suite (P1/2/4) + test_onnx.cpp (opt-in)
benchmarks/              Google Benchmark (opt-in, network at configure)
scripts/                 build-msvc.bat, make_test_model.py
```

## ONNX Runtime baseline (Phase 3)

Off by default (downloads the prebuilt ORT 1.17.3 win-x64 release, ~62 MB, at
configure time). The test model is regenerated with
`python scripts\make_test_model.py` (needs `pip install onnx numpy`) but the
`models/` asset is committed so the C++ side needs no Python.

```
cmake -S . -B build-onnx -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release ^
      -DSTRATA_BUILD_ONNX=ON -DSTRATA_BUILD_TESTS=ON
cmake --build build-onnx --target strata_onnx_test
build-onnx\strata_onnx_test.exe models
```

## Running the benchmarks

Benchmarks are off by default (the Google Benchmark fetch needs network at
configure time). Build into a separate tree:

```
cmake -S . -B build-bench -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release ^
      -DSTRATA_BUILD_BENCHMARKS=ON -DSTRATA_BUILD_TESTS=OFF
cmake --build build-bench --target bench_matvec
build-bench\benchmarks\bench_matvec.exe --benchmark_min_time=0.2s
```

## Phase 1 components

- **`Strata::MemoryPool`** — one 64-byte-aligned arena reserved at
  construction. `try_allocate()` is `noexcept`, heap-free, O(1) bump — the
  only carve permitted on the hot path. `reset()` recycles in O(1) and
  retains a high-water mark for arena sizing.
- **`Strata::Tensor<T>`** — non-owning, allocation-free (rank ≤ 8, inline
  metadata), row-major view onto pool memory. Every `create()` carve starts
  on a cache line. `std::span` views (`flat()`, `row()`), variadic indexing,
  and an `_mm_prefetch`-based `prefetch_row()` (portable fallback).

## Phase 2 components

`y = W·x` matrix-vector kernels, **allocation-free and `noexcept` by
contract** (nothing under `Strata::Compute` touches the heap).

- **Scalar reference** (`matvec_*_scalar`) — the correctness oracle and
  benchmark baseline; inner-loop auto-vectorisation is explicitly suppressed
  so the reported speed-up is real, not the optimiser vectorising for us.
- **AVX2/FMA kernels** (`matvec_*_avx2`) — FP32 via 8-wide
  `_mm256_fmadd_ps` + horizontal reduce; INT8 via `_mm256_cvtepi8_epi16` →
  `_mm256_madd_epi16` int32 accumulation then dequantise `Y = scale·(X·W)`.
  Unaligned loads are deliberate (kernels accept arbitrary caller buffers;
  cache-aligned Tensor data still fast-paths). Scalar-tail remainder handled.
- **`cpu_features()`** — one-time CPUID/XGETBV detection (AVX2/FMA/AVX-VNNI/
  AVX-512). Reports AVX-512 *false* on this CPU; it's the live seam.
- **Dispatcher** — `matvec_f32` / `matvec_i8` resolve `Backend::Auto` against
  the CPU once; `set_backend()` (lock-free atomic) pins Scalar/AVX2 for
  tests & benchmarks. `Backend::AVX512` is reserved and degrades to AVX2
  today, so a future AVX-512 kernel slots in without changing call sites.

Equivalence is tested (`test_compute.cpp`): AVX2 matches scalar within
tolerance for FP32 and **bit-exactly** for INT8, including K-remainder and
sub-vector-width paths.

## Phase 3 components

The spec's *dual-execution benchmark baseline* — ONNX Runtime via its C++
API, under `Strata::Onnx` (deliberately **not** `Strata::Compute`: ORT
manages its own memory, and the no-alloc contract is scoped to the custom
hot path, not the reference baseline).

- **`Strata::Onnx::OnnxModel`** — pImpl wrapper so `<onnxruntime_cxx_api.h>`
  never leaks into the public surface. Loads a serialised `.onnx`, exposes
  I/O names + static shapes + element counts, and runs a single-input/output
  float graph. `std::filesystem::path::c_str()` gives the right `ORTCHAR_T`
  on both Windows (`wchar_t`) and POSIX (`char`).
- **`cmake/onnxruntime.cmake`** — fetches Microsoft's official prebuilt ORT
  release and exposes it as an imported target; the runtime DLL is
  auto-copied next to the test exe.
- **Test model** — `scripts/make_test_model.py` builds a 2-layer MLP
  (Gemm→Relu→Gemm) with fixed-seed weights and emits a numpy-computed
  input/output oracle. `test_onnx.cpp` verifies introspection, that the ORT
  output matches the oracle within 1e-4, and that re-runs are bit-identical.

This baseline is what Phase 5 will time head-to-head against the custom
StrataCompute forward pass.

## Phase 4 components

The custom engine — the whole stack composed, **ORT-free and offline**.

- **`Strata::Onnx::OnnxGraph`** — a tight, dependency-free reader of just the
  ONNX protobuf subset Phase 4 needs (ModelProto → GraphProto → node /
  initializer / value-info). ORT's inference API doesn't expose initializers
  and pulling protobuf is heavy, so we decode the wire format directly
  (~150 LOC, generic field-skipping). Float `raw_data` initializers only.
- **`Strata::Engine::ForwardEngine`** — at construction: extracts every float
  initializer into `Strata::Tensor`s backed by a weights `MemoryPool`, then
  compiles the node list (Gemm transB=1 + optional bias, Relu) into a flat
  execution plan with operands pre-resolved to a tagged input/output/slot
  ref. `forward()`: `reset()`s the activation arena, bump-carves one buffer
  per intermediate, and runs the plan through the Phase 2
  `Strata::Compute::matvec_f32` (AVX2) + `add_inplace` + `relu` operators.
  **The forward hot path performs zero heap allocations.**
- **`Strata::Compute::{add_inplace,relu,relu_inplace}`** — the small
  allocation-free elementwise ops between matvec layers.

`test_engine.cpp` proves the composition: engine output matches the same
numpy oracle ONNX Runtime was validated against in Phase 3 (so
engine ≡ ORT transitively), 1000 inferences through the recycled arena are
bit-identical, and wrong input sizes are rejected. Supported op set is the
spec's feed-forward MLP family (Gemm/Relu); weight extraction is generic.

## Phase 5 components

End-to-end three-way latency comparison on the *same* model and input.

- **`benchmarks/bench_pipeline.cpp`** — times ONNX Runtime (C++) vs
  StrataCompute with per-inference (not block-averaged) wall timing so the
  P95/P99 **tail** is real; gates on a correctness check vs the oracle before
  timing; writes `results/latency_cpp.csv`. Built only in the ONNX tree.
- **`scripts/run_phase5.py`** — builds the *same* MLP in PyTorch (CPU,
  single-thread) from the weights extracted out of `mlp.onnx`, times it,
  merges the C++ CSV, and renders `results/latency.csv` +
  [docs/benchmarks.md](docs/benchmarks.md).
- **Cache misses** — not fabricated: hardware L1/L2 counters need Linux
  `perf`/Cachegrind/Intel PCM (absent on this Windows+MSVC box, as the spec's
  own Cachegrind reference implies). The report instead gives the **exact**
  analytical working set the custom engine touches (~3.5 KiB, L1/L2-resident
  — the structural reason the tail is flat) plus the Linux commands for real
  counters.

Run it all (needs the ONNX tree + `pip install torch onnx numpy`):

```
scripts\run-phase5.bat
```
