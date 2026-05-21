// Phase 2 — Strata::Compute kernel correctness.
//
//  Strategy: the scalar kernel is the oracle. The AVX2 kernel must match it —
//  exactly for INT8 (identical int32 accumulation), and within a tight
//  tolerance for FP32 (FMA + different summation order perturb rounding).
//  Buffers come from Strata::Tensor/MemoryPool so Phase 1 is exercised too,
//  including the K-remainder paths (K not a multiple of 8 or 16).
#include "strata/compute/matvec.hpp"
#include "strata/compute/cpu_features.hpp"
#include "strata/tensor.hpp"
#include "strata/memory_pool.hpp"
#include "test_harness.hpp"

#include <cmath>
#include <cstdint>

using Strata::MemoryPool;
using Strata::Tensor;
namespace C = Strata::Compute;

namespace {

// Deterministic LCG so failures reproduce.
struct Rng {
    std::uint64_t s = 0x9E3779B97F4A7C15ull;
    float next_f() {              // [-1, 1)
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<float>(static_cast<std::int32_t>(s >> 33)) /
               2147483648.0f;
    }
    std::int8_t next_i8() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<std::int8_t>((s >> 40) & 0xFF);
    }
};

} // namespace

STRATA_TEST(f32_avx2_matches_scalar_with_tail) {
    constexpr std::size_t M = 11, K = 37;     // K%8 == 5 -> remainder path
    MemoryPool pool(1 << 16);
    auto W  = Tensor<float>::create(pool, {M, K});
    auto x  = Tensor<float>::create(pool, {K});
    auto ya = Tensor<float>::create(pool, {M});
    auto yb = Tensor<float>::create(pool, {M});

    Rng r;
    for (auto& v : W.flat()) v = r.next_f();
    for (auto& v : x.flat()) v = r.next_f();

    C::kernels::matvec_f32_scalar(ya.data(), W.data(), x.data(), M, K);
    C::kernels::matvec_f32_avx2(yb.data(), W.data(), x.data(), M, K);

    for (std::size_t i = 0; i < M; ++i) {
        STRATA_CHECK(std::fabs(ya(i) - yb(i)) < 1e-3f);
    }
}

STRATA_TEST(f32_avx2_handles_K_below_vector_width) {
    constexpr std::size_t M = 4, K = 3;       // pure scalar tail, no full lane
    MemoryPool pool(1 << 12);
    auto W  = Tensor<float>::create(pool, {M, K});
    auto x  = Tensor<float>::create(pool, {K});
    auto ya = Tensor<float>::create(pool, {M});
    auto yb = Tensor<float>::create(pool, {M});
    Rng r;
    for (auto& v : W.flat()) v = r.next_f();
    for (auto& v : x.flat()) v = r.next_f();

    C::kernels::matvec_f32_scalar(ya.data(), W.data(), x.data(), M, K);
    C::kernels::matvec_f32_avx2(yb.data(), W.data(), x.data(), M, K);
    for (std::size_t i = 0; i < M; ++i)
        STRATA_CHECK(std::fabs(ya(i) - yb(i)) < 1e-4f);
}

STRATA_TEST(i8_avx2_exactly_matches_scalar) {
    constexpr std::size_t M = 9, K = 53;      // K%16 == 5 -> remainder path
    constexpr float scale = 0.0125f;
    MemoryPool pool(1 << 16);
    auto W  = Tensor<std::int8_t>::create(pool, {M, K});
    auto x  = Tensor<std::int8_t>::create(pool, {K});
    auto ya = Tensor<float>::create(pool, {M});
    auto yb = Tensor<float>::create(pool, {M});
    Rng r;
    for (auto& v : W.flat()) v = r.next_i8();
    for (auto& v : x.flat()) v = r.next_i8();

    C::kernels::matvec_i8_scalar(ya.data(), W.data(), x.data(), M, K, scale);
    C::kernels::matvec_i8_avx2(yb.data(), W.data(), x.data(), M, K, scale);

    // Identical int32 accumulation -> bit-identical dequantised output.
    for (std::size_t i = 0; i < M; ++i)
        STRATA_CHECK_EQ(ya(i), yb(i));
}

STRATA_TEST(i8_dequant_value_is_correct) {
    // Hand-checkable: W row = {1,2,3,4}, x = {10,10,10,10}, scale = 0.5
    // dot = (1+2+3+4)*10 = 100 ; y = 0.5 * 100 = 50
    constexpr std::size_t M = 1, K = 4;
    MemoryPool pool(1 << 12);
    auto W = Tensor<std::int8_t>::create(pool, {M, K});
    auto x = Tensor<std::int8_t>::create(pool, {K});
    auto y = Tensor<float>::create(pool, {M});
    for (std::size_t j = 0; j < K; ++j) { W(0, j) = static_cast<std::int8_t>(j + 1); x(j) = 10; }

    C::kernels::matvec_i8_scalar(y.data(), W.data(), x.data(), M, K, 0.5f);
    STRATA_CHECK_EQ(y(0), 50.0f);
    C::kernels::matvec_i8_avx2(y.data(), W.data(), x.data(), M, K, 0.5f);
    STRATA_CHECK_EQ(y(0), 50.0f);
}

STRATA_TEST(dispatcher_set_backend_round_trips) {
    const auto& cf = C::cpu_features();
    STRATA_CHECK(cf.sse2);                       // any x64 CPU has SSE2

    C::set_backend(C::Backend::Scalar);
    STRATA_CHECK_EQ(C::active_backend(), C::Backend::Scalar);

    C::set_backend(C::Backend::AVX2);
    // Resolves to AVX2 on this CPU (has AVX2+FMA), else degrades to Scalar.
    const C::Backend expect =
        (cf.avx2 && cf.fma) ? C::Backend::AVX2 : C::Backend::Scalar;
    STRATA_CHECK_EQ(C::active_backend(), expect);

    C::set_backend(C::Backend::AVX512);          // reserved seam -> best avail
    STRATA_CHECK_EQ(C::active_backend(), expect);

    C::set_backend(C::Backend::Auto);            // restore default dispatch
    STRATA_CHECK_EQ(C::active_backend(), expect);
}

STRATA_TEST(dispatched_matvec_matches_pinned_kernels) {
    constexpr std::size_t M = 6, K = 40;
    MemoryPool pool(1 << 16);
    auto W   = Tensor<float>::create(pool, {M, K});
    auto x   = Tensor<float>::create(pool, {K});
    auto ref = Tensor<float>::create(pool, {M});
    auto got = Tensor<float>::create(pool, {M});
    Rng r;
    for (auto& v : W.flat()) v = r.next_f();
    for (auto& v : x.flat()) v = r.next_f();

    C::kernels::matvec_f32_scalar(ref.data(), W.data(), x.data(), M, K);

    C::set_backend(C::Backend::AVX2);
    C::matvec_f32(got.flat(), W.data(), x.data(), M, K);
    for (std::size_t i = 0; i < M; ++i)
        STRATA_CHECK(std::fabs(ref(i) - got(i)) < 1e-3f);

    C::set_backend(C::Backend::Scalar);
    C::matvec_f32(got.flat(), W.data(), x.data(), M, K);
    for (std::size_t i = 0; i < M; ++i)
        STRATA_CHECK_EQ(ref(i), got(i));         // same kernel -> exact

    C::set_backend(C::Backend::Auto);
}
