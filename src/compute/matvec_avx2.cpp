// =============================================================================
//  Strata::Compute — hand-rolled AVX2 / FMA kernels
// -----------------------------------------------------------------------------
//  FP32 : 8-wide _mm256_fmadd_ps accumulation + horizontal reduce, scalar tail.
//  INT8 : 16 int8 / iter — sign-extend to int16 (_mm256_cvtepi8_epi16),
//         pairwise multiply-add (_mm256_madd_epi16) into an int32 accumulator
//         that exactly mirrors the scalar reference, then dequantise.
//
//  Unaligned loads (_mm256_loadu_ps / _mm_loadu_si128) are used deliberately:
//  the kernel accepts arbitrary caller buffers, and an aligned _mm256_load_ps
//  would #GP-fault on a non-64B pointer. On Tensor-sourced (cache-aligned)
//  data the hardware fast-paths loadu to the same throughput as load.
//
//  Allocation-free, noexcept — honours the Strata::Compute contract.
// =============================================================================
#include "strata/compute/matvec.hpp"

#include <immintrin.h>
#include <cstdint>

namespace Strata::Compute::kernels {
namespace {

[[nodiscard]] float hsum256_ps(__m256 v) noexcept {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);     // {a,b,c,d}
    lo = _mm_hadd_ps(lo, lo);    // {a+b, c+d, ...}
    lo = _mm_hadd_ps(lo, lo);    // {a+b+c+d, ...}
    return _mm_cvtss_f32(lo);
}

[[nodiscard]] std::int32_t hsum256_epi32(__m256i v) noexcept {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s  = _mm_add_epi32(lo, hi);
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
}

} // namespace

void matvec_f32_avx2(float* y, const float* W, const float* x,
                     std::size_t M, std::size_t K) noexcept {
    const std::size_t Kv = K & ~std::size_t{7};  // largest multiple of 8
    for (std::size_t i = 0; i < M; ++i) {
        const float* w = W + i * K;
        __m256 acc = _mm256_setzero_ps();
        for (std::size_t j = 0; j < Kv; j += 8) {
            const __m256 wv = _mm256_loadu_ps(w + j);
            const __m256 xv = _mm256_loadu_ps(x + j);
            acc = _mm256_fmadd_ps(wv, xv, acc);   // w*x + acc, single rounding
        }
        float s = hsum256_ps(acc);
        for (std::size_t j = Kv; j < K; ++j) {    // remainder (K % 8)
            s += w[j] * x[j];
        }
        y[i] = s;
    }
}

void matvec_i8_avx2(float* y, const std::int8_t* W, const std::int8_t* x,
                    std::size_t M, std::size_t K, float scale) noexcept {
    const std::size_t Kv = K & ~std::size_t{15};  // largest multiple of 16
    for (std::size_t i = 0; i < M; ++i) {
        const std::int8_t* w = W + i * K;
        __m256i acc = _mm256_setzero_si256();
        for (std::size_t j = 0; j < Kv; j += 16) {
            const __m128i w8 =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + j));
            const __m128i x8 =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + j));
            const __m256i w16 = _mm256_cvtepi8_epi16(w8);   // 16x int8 -> int16
            const __m256i x16 = _mm256_cvtepi8_epi16(x8);
            const __m256i prod = _mm256_madd_epi16(w16, x16); // 8x int32
            acc = _mm256_add_epi32(acc, prod);
        }
        std::int32_t s = hsum256_epi32(acc);
        for (std::size_t j = Kv; j < K; ++j) {    // remainder (K % 16)
            s += static_cast<std::int32_t>(w[j]) *
                 static_cast<std::int32_t>(x[j]);
        }
        y[i] = scale * static_cast<float>(s);     // dequantise
    }
}

} // namespace Strata::Compute::kernels
