// =============================================================================
//  Strata::Compute  —  hand-rolled matrix-vector kernels
// -----------------------------------------------------------------------------
//  y = W * x   where W is row-major [M x K], x is [K], y is [M].
//
//  Two precisions:
//    * FP32          : y[i] = sum_j W[i,j] * x[j]
//    * INT8 quantized : y[i] = scale * sum_j (int32) W[i,j] * x[j]
//                       (the dequantisation Y = scale * (X . W))
//
//  Hard contract (Phase-2 directive): **no dynamic allocation anywhere under
//  Strata::Compute**. Every entry point is noexcept and operates only on
//  caller-owned buffers — no new/delete, no std::vector, no scratch heap.
//
//  Backend model
//  -------------
//  The dispatched `matvec_*` functions select a kernel from
//  `cpu_features()` once. Explicit `kernels::*_scalar` / `*_avx2` entry
//  points are exposed so the equivalence tests and Google Benchmark cases can
//  pin a specific implementation. `set_backend()` is the override seam an
//  AVX-512 kernel will plug into later without changing call sites.
//
//  ISA-baseline note: the Release build sets /arch:AVX2 globally (CMake),
//  so this binary already requires AVX2. The dispatcher's present value is
//  (a) honest kernel selection AVX2-vs-scalar for benchmarking, and
//  (b) the structural seam for a future per-TU AVX-512 kernel.
// =============================================================================
#ifndef STRATA_COMPUTE_MATVEC_HPP
#define STRATA_COMPUTE_MATVEC_HPP

#include <cstddef>
#include <cstdint>
#include <span>

namespace Strata::Compute {

enum class Backend {
    Auto,    // pick best from cpu_features() (default)
    Scalar,  // portable reference
    AVX2,    // hand-rolled AVX2/FMA
    AVX512   // reserved seam — falls back to AVX2/Scalar until implemented
};

/// Override kernel selection process-wide (test/benchmark seam). Auto restores
/// feature-based dispatch. Allocation-free, noexcept.
void    set_backend(Backend b) noexcept;
/// The concrete backend the dispatcher will run for the current selection.
[[nodiscard]] Backend active_backend() noexcept;

// ---- Dispatched entry points ----------------------------------------------
/// y[0..M) = W[M x K] (row-major) * x[0..K).  y.size() must be >= M.
void matvec_f32(std::span<float> y, const float* W, const float* x,
                std::size_t M, std::size_t K) noexcept;

/// y[i] = scale * sum_j (int32) W[i,j] * x[j].  Signed int8 inputs.
void matvec_i8(std::span<float> y, const std::int8_t* W, const std::int8_t* x,
               std::size_t M, std::size_t K, float scale) noexcept;

// ---- Pinned kernels (for tests / benchmarks) ------------------------------
namespace kernels {

void matvec_f32_scalar(float* y, const float* W, const float* x,
                        std::size_t M, std::size_t K) noexcept;
void matvec_f32_avx2(float* y, const float* W, const float* x,
                     std::size_t M, std::size_t K) noexcept;

void matvec_i8_scalar(float* y, const std::int8_t* W, const std::int8_t* x,
                      std::size_t M, std::size_t K, float scale) noexcept;
void matvec_i8_avx2(float* y, const std::int8_t* W, const std::int8_t* x,
                    std::size_t M, std::size_t K, float scale) noexcept;

} // namespace kernels

} // namespace Strata::Compute

#endif // STRATA_COMPUTE_MATVEC_HPP
