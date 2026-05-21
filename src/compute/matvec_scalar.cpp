// =============================================================================
//  Strata::Compute — portable scalar reference kernels
// -----------------------------------------------------------------------------
//  These are the correctness oracle *and* the benchmark baseline. Compiler
//  auto-vectorisation of the inner reduction is explicitly suppressed so the
//  "scalar vs hand-rolled SIMD" speed-up reported in Phase 2 is genuine and
//  not just the optimiser vectorising this loop for us under /arch:AVX2.
// =============================================================================
#include "strata/compute/matvec.hpp"

#include <cstdint>

// Disable inner-loop auto-vectorisation (keep -O2/-O3 elsewhere).
#if defined(_MSC_VER)
#  define STRATA_NOVEC __pragma(loop(no_vector))
#elif defined(__clang__)
#  define STRATA_NOVEC _Pragma("clang loop vectorize(disable)")
#else
#  define STRATA_NOVEC
#endif

namespace Strata::Compute::kernels {

void matvec_f32_scalar(float* y, const float* W, const float* x,
                        std::size_t M, std::size_t K) noexcept {
    for (std::size_t i = 0; i < M; ++i) {
        const float* w = W + i * K;
        float acc = 0.0f;
        STRATA_NOVEC
        for (std::size_t j = 0; j < K; ++j) {
            acc += w[j] * x[j];
        }
        y[i] = acc;
    }
}

void matvec_i8_scalar(float* y, const std::int8_t* W, const std::int8_t* x,
                       std::size_t M, std::size_t K, float scale) noexcept {
    for (std::size_t i = 0; i < M; ++i) {
        const std::int8_t* w = W + i * K;
        std::int32_t acc = 0;  // exact integer accumulation, then dequantise
        STRATA_NOVEC
        for (std::size_t j = 0; j < K; ++j) {
            acc += static_cast<std::int32_t>(w[j]) *
                   static_cast<std::int32_t>(x[j]);
        }
        y[i] = scale * static_cast<float>(acc);
    }
}

} // namespace Strata::Compute::kernels
