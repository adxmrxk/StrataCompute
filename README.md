# StrataCompute

**Bare-metal C++20 ML inference engine: deterministic, sub-microsecond tail latency.**

StrataCompute is a from-scratch ML inference engine that bypasses the Python runtime, owns every byte of its memory, hand-rolls its SIMD linear algebra, and decodes ONNX protobufs directly. On a single-sample MLP it runs **172× faster in the mean and 425× faster at the P99 tail than PyTorch**, and **57× faster than ONNX Runtime's C++ API**, all while producing output that is bit-close to a numpy oracle.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Headline Latency Numbers](#headline-latency-numbers)
- [The Five Phases](#the-five-phases)
- [Key Engineering Decisions](#key-engineering-decisions)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Docker Build](#docker-build)
- [Kubernetes Deployment](#kubernetes-deployment)

---

## Project Overview

### What It Is

A single-process, single-thread, statically-linked C++20 library that loads a small ONNX model into custom memory pools, extracts its weights into cache-aligned tensors, and runs forward passes through hand-written AVX2 + FMA kernels. There is no Python anywhere on the hot path. There is no heap allocation anywhere on the hot path. The execution plan is a flat array of operators dispatched without virtual calls.

The goal is not throughput, it is **tail latency**. Most ML inference systems care about how many samples per second they can serve on a hot GPU. StrataCompute cares about how many microseconds the slowest 1% of single-sample inferences take on a cold cache, because that is the number that matters for systems where the next decision has to be made before the next packet arrives.

### The Problem It Solves

PyTorch and ONNX Runtime are excellent general-purpose runtimes, but they are optimized for the average case. Both carry meaningful per-inference overhead: PyTorch from its dynamic dispatch and Python interop layer, ONNX Runtime from its graph manager and memory planner. For a tiny MLP-sized model running a single sample at a time, that overhead dominates the actual arithmetic.

StrataCompute eliminates the overhead by doing every expensive thing **once**: extract weights once at construction, compile the execution plan once, reserve the activation arena once. The forward pass itself then becomes nothing but matrix-vector multiplies and elementwise ops on memory that was already laid out in cache lines exactly the way the kernels want it.

---

## Headline Latency Numbers

Measured on an Intel i5-12400 (Alder Lake), Windows MSVC `/O2 /arch:AVX2`, single-sample MLP, single thread, identical weights and inputs across all three engines.

| Engine                  | Mean       | P99        |
|-------------------------|-----------:|-----------:|
| PyTorch (Python)        | 20.97 µs   | 84.90 µs   |
| ONNX Runtime (C++)      | 6.99 µs    | 19.90 µs   |
| **StrataCompute (C++)** | **0.12 µs** | **0.20 µs** |

- ~172× lower mean and ~425× lower P99 tail vs PyTorch
- ~57× vs ONNX Runtime
- Output verified bit-close to the numpy oracle (max absolute error 1.8e-7)

### Phase 2 measured SIMD acceleration

Hand-rolled AVX2 + FMA versus the explicitly auto-vectorization-suppressed scalar reference:

| Kernel              | 256² | 512² | 1024² |
|---------------------|-----:|-----:|------:|
| FP32 scalar → AVX2  | 6.7× | 6.7× |  5.9× |
| INT8 scalar → AVX2  | 7.0× | 8.4× |  9.0× |

INT8 AVX2 sustains ~28 G MAC/s. FP32 AVX2 sustains ~12–19 G/s.

Full benchmark tables and the latency CSV are in [docs/benchmarks.md](docs/benchmarks.md).

---

## The Five Phases

StrataCompute was built phase-by-phase, with each phase as a self-contained deliverable that passes its own test suite under `/W4 /WX`.

### Phase 1 — Memory and Tensor Primitives

- **`Strata::MemoryPool`** — one 64-byte-aligned arena reserved at construction. `try_allocate()` is `noexcept`, heap-free, O(1) bump allocator. `reset()` recycles in O(1) and retains a high-water mark for arena sizing. This is the only memory carve permitted on the hot path.
- **`Strata::Tensor<T>`** — non-owning, allocation-free, row-major view onto pool memory (rank ≤ 8 with inline metadata). Every `create()` carve starts on a cache line. Provides `std::span` views (`flat()`, `row()`), variadic indexing, and a portable `_mm_prefetch` wrapper.

### Phase 2 — Hand-Rolled SIMD Kernels

`y = W·x` matrix-vector kernels. **Allocation-free and `noexcept` by contract.** Nothing under `Strata::Compute` touches the heap.

- **Scalar reference** (`matvec_*_scalar`) — correctness oracle and benchmark baseline. Inner-loop auto-vectorization is explicitly suppressed so the reported speedup is real, not the optimizer vectorizing for us.
- **AVX2 + FMA kernels** (`matvec_*_avx2`) — FP32 via 8-wide `_mm256_fmadd_ps` plus horizontal reduce. INT8 via `_mm256_cvtepi8_epi16` → `_mm256_madd_epi16` int32 accumulation, then dequantize `Y = scale·(X·W)`. Unaligned loads are deliberate so the kernels accept arbitrary caller buffers; cache-aligned `Tensor` data still fast-paths. Scalar-tail remainder handled correctly.
- **`cpu_features()`** — one-time CPUID + XGETBV detection (AVX2 / FMA / AVX-VNNI / AVX-512). Reports AVX-512 false on consumer Alder Lake; this is the live extension seam.
- **Dispatcher** — `matvec_f32` / `matvec_i8` resolve `Backend::Auto` against the CPU once. `set_backend()` (lock-free atomic) pins scalar or AVX2 for tests and benchmarks. `Backend::AVX512` is reserved and degrades to AVX2 today; a future AVX-512 kernel slots in without touching call sites.

### Phase 3 — ONNX Runtime Baseline

The reference baseline that Phase 5 times against.

- **`Strata::Onnx::OnnxModel`** — pImpl wrapper so `<onnxruntime_cxx_api.h>` never leaks into the public surface. Loads a serialized `.onnx`, exposes I/O names + static shapes + element counts, and runs a single-input/output float graph. `std::filesystem::path::c_str()` gives the right `ORTCHAR_T` on both Windows (`wchar_t`) and POSIX (`char`).
- **`cmake/onnxruntime.cmake`** — fetches Microsoft's official prebuilt ORT release and exposes it as an imported target. The runtime DLL is auto-copied next to the test exe.
- **Test model** — `scripts/make_test_model.py` builds a 2-layer MLP (Gemm → Relu → Gemm) with fixed-seed weights and emits a numpy-computed input/output oracle.

### Phase 4 — Custom Forward Pass

The whole stack composed. **ORT-free and offline.**

- **`Strata::Onnx::OnnxGraph`** — a tight, dependency-free reader of just the ONNX protobuf subset needed (ModelProto → GraphProto → node / initializer / value-info). ORT's inference API doesn't expose initializers and pulling in full protobuf is heavy, so we decode the wire format directly (~150 LOC, generic field-skipping). Float `raw_data` initializers only.
- **`Strata::Engine::ForwardEngine`** — at construction, extracts every float initializer into `Strata::Tensor`s backed by a weights `MemoryPool`, then compiles the node list into a flat execution plan with operands pre-resolved to tagged input/output/slot refs. `forward()` calls `reset()` on the activation arena, bump-carves one buffer per intermediate, and runs the plan through the Phase 2 kernels. **The forward hot path performs zero heap allocations.**
- **`Strata::Compute::{add_inplace, relu, relu_inplace}`** — the small allocation-free elementwise ops between matvec layers.

`test_engine.cpp` proves the composition: engine output matches the same numpy oracle ONNX Runtime was validated against in Phase 3, 1000 inferences through the recycled arena are bit-identical, and wrong input sizes are rejected.

### Phase 5 — Microsecond Benchmarking

End-to-end three-way latency comparison on the same model and input.

- **`benchmarks/bench_pipeline.cpp`** — times ONNX Runtime (C++) vs StrataCompute with per-inference (not block-averaged) wall timing so the P95 and P99 tail are real. Gates on a correctness check vs the oracle before timing. Writes `results/latency_cpp.csv`.
- **`scripts/run_phase5.py`** — builds the same MLP in PyTorch (CPU, single thread) from the weights extracted out of `mlp.onnx`, times it, merges the C++ CSV, and renders `results/latency.csv` plus [docs/benchmarks.md](docs/benchmarks.md).
- **Cache misses** — hardware L1/L2 counters need Linux `perf`, Cachegrind, or Intel PCM. The report gives the exact analytical working set the custom engine touches (~3.5 KiB, fully L1/L2-resident, which is the structural reason the tail is flat) plus the Linux commands for real counters. See [Docker Build](#docker-build) for the Cachegrind path.

---

## Key Engineering Decisions

### Allocation-free hot path

Every allocation happens at construction. The forward pass calls `MemoryPool::reset()` and bump-allocates activation buffers into the arena. There is no `new`, no `malloc`, no `std::vector::resize`, no `string` operation on the hot path.

### No virtual calls

The execution plan is a flat array of operator descriptors. Dispatch is a switch over op kind, not a vtable. The branch predictor learns the pattern after the first few inferences and dispatch becomes essentially free.

### Auto-vectorization is suppressed on the scalar reference

A 6× speedup over the scalar baseline is only meaningful if the scalar baseline is actually scalar. The scalar kernels disable auto-vectorization on their inner loops so the AVX2 speedup is real, not the optimizer vectorizing the reference for us.

### Direct ONNX protobuf decode

Linking in the full protobuf library to read a few model fields adds binary size, configure-time cost, and a runtime dependency. The graph reader hand-decodes the protobuf wire format for just the messages needed (ModelProto, GraphProto, NodeProto, TensorProto) in about 150 lines of code.

### Toolchain dual targeting

The original spec assumed GCC + Linux + AVX-512. The actual dev machine forced MSVC + Windows + AVX2. The CMake flag abstraction retains both the GCC `-O3 -march=native -flto -Wall -Wextra -Werror` path and the MSVC `/O2 /GL /W4 /WX /std:c++20 /arch:AVX2` path; the same source tree builds correctly on either.

### AVX-512 is an empty seam

The dispatcher has an `AVX512` backend that degrades to AVX2 today. When this project runs on hardware where Intel hasn't fused AVX-512 off (server parts, or 11th-gen and earlier consumer), a future AVX-512 kernel slots in by implementing one function and updating the dispatcher table.

---

## Tech Stack

| Component                | Notes |
|--------------------------|-------|
| **Language**             | C++20 |
| **Build system**         | CMake 3.20+ |
| **Compilers tested**     | MSVC 19.29 (`/O2 /GL /W4 /WX /std:c++20 /arch:AVX2`) and GCC (`-O3 -march=native -flto -Wall -Wextra -Wpedantic -Werror`) |
| **Testing**              | Custom lightweight harness + Google Benchmark for Phase 2/5 |
| **ONNX Runtime**         | Microsoft's prebuilt release 1.17.3, fetched at configure time |
| **PyTorch (Phase 5)**    | CPU build, single-thread, used only for the latency comparison |
| **SIMD**                 | AVX2 + FMA. AVX-512 / AVX-VNNI detection live, kernels reserved |
| **Containerization**     | Docker (GCC `-O3 -march=native` + Valgrind / Cachegrind) |
| **Orchestration**        | Kubernetes `Job` and `CronJob` for batch latency-drift watch |

---

## Project Structure

```
stratacompute/
│
├── include/strata/
│   ├── memory_pool.hpp                MemoryPool: bump arena, 64-byte aligned
│   ├── tensor.hpp                     Tensor<T>: non-owning view, rank ≤ 8
│   ├── compute/
│   │   ├── matvec.hpp                 matvec_f32 / matvec_i8 with backend enum
│   │   ├── elementwise.hpp            add_inplace, relu, relu_inplace
│   │   └── cpu_features.hpp           CPUID + XGETBV detection
│   ├── onnx/
│   │   ├── onnx_model.hpp             pImpl ORT wrapper
│   │   └── onnx_graph.hpp             Custom protobuf reader
│   └── engine/
│       └── forward_engine.hpp         Custom pure-C++ forward pass
│
├── src/
│   ├── memory_pool.cpp
│   ├── compute/
│   │   ├── cpu_features.cpp
│   │   ├── matvec_scalar.cpp          Reference baseline (auto-vec suppressed)
│   │   ├── matvec_avx2.cpp            Hand-written FP32 + INT8 kernels
│   │   ├── matvec_dispatch.cpp        Backend resolution
│   │   └── elementwise.cpp
│   ├── onnx/
│   │   ├── onnx_model.cpp             ORT wrapper implementation
│   │   └── onnx_graph.cpp             Protobuf decoder (~150 LOC)
│   └── engine/
│       └── forward_engine.cpp         Compile + run flat execution plan
│
├── benchmarks/
│   ├── bench_matvec.cpp               Phase 2 kernel benchmarks
│   ├── bench_memory_pool.cpp
│   └── bench_pipeline.cpp             Phase 5 latency comparison
│
├── tests/
│   ├── test_main.cpp                  Custom harness
│   ├── test_memory_pool.cpp
│   ├── test_tensor.cpp
│   ├── test_compute.cpp               Scalar vs AVX2 equivalence
│   ├── test_onnx.cpp                  Phase 3 ORT vs numpy oracle
│   └── test_engine.cpp                Phase 4 custom forward pass
│
├── cmake/
│   └── onnxruntime.cmake              Prebuilt ORT fetcher
│
├── scripts/
│   ├── make_test_model.py             Generate mlp.onnx + oracle
│   ├── run_phase5.py                  Three-way latency runner
│   ├── run-phase5.bat
│   └── docker-entrypoint.sh           test / phase5 / cachegrind
│
├── k8s/                               Job + CronJob manifests (see below)
│   ├── job.yaml
│   ├── cronjob.yaml
│   ├── namespace.yaml
│   ├── pvc.yaml
│   └── kustomization.yaml
│
├── models/                            mlp.onnx + mlp_io.csv (committed)
├── results/                           Benchmark output CSVs
├── docs/benchmarks.md                 Rendered Phase 5 results
│
├── CMakeLists.txt
├── Dockerfile
└── README.md
```

---

## Getting Started

### Windows / MSVC

VS2019 Build Tools is the Build Tools SKU (no IDE), which CMake's Visual Studio generator can't auto-discover via `vswhere`. Drive the build through `vcvars64` plus the NMake Makefiles generator:

```bat
scripts\build-msvc.bat Release
```

Manual equivalent (from a `vcvars64`-initialized shell):

```bat
cmake -S . -B build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

### Phase 3 ONNX baseline (opt-in)

Off by default; downloads the prebuilt ONNX Runtime 1.17.3 win-x64 release (~62 MB) at configure time:

```bat
cmake -S . -B build-onnx -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release ^
      -DSTRATA_BUILD_ONNX=ON -DSTRATA_BUILD_TESTS=ON
cmake --build build-onnx --target strata_onnx_test
build-onnx\strata_onnx_test.exe models
```

### Phase 2 benchmarks (opt-in)

Google Benchmark is fetched at configure time, so the suite needs network:

```bat
cmake -S . -B build-bench -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release ^
      -DSTRATA_BUILD_BENCHMARKS=ON -DSTRATA_BUILD_TESTS=OFF
cmake --build build-bench --target bench_matvec
build-bench\benchmarks\bench_matvec.exe --benchmark_min_time=0.2s
```

### Phase 5 latency comparison

```bat
scripts\run-phase5.bat
```

Needs the ONNX tree built plus `pip install torch onnx numpy`.

---

## Docker Build

The Docker image realizes the spec's original intended toolchain: GCC with `-O3 -march=native -flto -Wall -Wextra -Wpedantic -Werror`, C++20. Includes Valgrind, so the real L1/LL cache-miss counts the spec asked for (impossible to capture on Windows) can be measured.

```sh
docker build -t stratacompute .                                       # builds + runs P1/2/4 (ctest gate)
docker run --rm stratacompute                                          # test + phase5 + cachegrind
docker run --rm stratacompute test                                     # just the suites
docker run --rm -v "$PWD/out:/app/out" stratacompute phase5            # collect reports
docker run --rm -v "$PWD/out:/app/out" stratacompute cachegrind        # real cache misses
```

The offline suite (Phases 1, 2, 4) is compiled **and run** during `docker build`, so the image fails to build if anything regresses under GCC `-Werror`. The build is its own verification gate.

---

## Kubernetes Deployment

StrataCompute is a **batch workload**: it runs the suite + Phase 5 benchmark + Cachegrind to completion and exits. The correct primitives are a `Job` (one-shot run) and a `CronJob` (scheduled latency-drift watch), not a `Deployment` / `Service`. There is no server, no port, nothing to load-balance. A Deployment would just crash-loop on the process exiting.

```sh
# Build and push the image first (or load it into kind/minikube).
# Edit the images: block in k8s/kustomization.yaml to point at your registry.

kubectl apply -k k8s/
kubectl -n stratacompute logs job/stratacompute-bench -f
kubectl -n stratacompute cp <pod>:/app/out ./out      # pull reports
kubectl delete -k k8s/
```

The manifests encode several things that matter for HFT-grade latency work:

- **Guaranteed QoS** (`requests == limits`, integer CPU) so the pod is eligible for the kubelet's static CPU Manager. Sub-microsecond P99 is only trustworthy with `--cpu-manager-policy=static`, CFS throttling off, and NUMA-local isolated cores on the node.
- **`-march=native` portability** — the image is compiled for the build host's CPU; pin nodes via `nodeSelector` / affinity to a matching CPU, or rebuild with a fixed `-march` for heterogeneous clusters, otherwise SIGILL.
- `restartPolicy: Never` + `backoffLimit: 1` so a failed correctness gate surfaces instead of silently retrying. PVC retains reports after pod exit.
