// =============================================================================
//  Phase 2 — Google Benchmark: hand-rolled SIMD vs scalar reference.
//
//  Proves the AVX2/FMA matrix-vector kernels accelerate over the portable
//  scalar reference, for both FP32 and INT8-quantised paths, across layer
//  sizes typical of an MLP/ResNet FC stage. Buffers come from a
//  Strata::MemoryPool (no heap traffic in the timed region — same discipline
//  as the engine hot path).
//
//  Read the result as the Scalar/AVX2 time ratio for matched {M,K}; the
//  `items_per_second` (= MACs/s) column makes the speed-up explicit.
// =============================================================================
#include "strata/compute/matvec.hpp"
#include "strata/memory_pool.hpp"
#include "strata/tensor.hpp"

#include <benchmark/benchmark.h>

#include <cstdint>

using Strata::MemoryPool;
using Strata::Tensor;
namespace C = Strata::Compute;

namespace {

void fill_f32(Tensor<float>& t, std::uint64_t seed) {
    std::uint64_t s = seed;
    for (auto& v : t.flat()) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        v = static_cast<float>(static_cast<std::int32_t>(s >> 33)) / 2.1e9f;
    }
}
void fill_i8(Tensor<std::int8_t>& t, std::uint64_t seed) {
    std::uint64_t s = seed;
    for (auto& v : t.flat()) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        v = static_cast<std::int8_t>((s >> 40) & 0xFF);
    }
}

// Arena big enough for W[M*K] + x[K] + y[M] at the largest benched size.
constexpr std::size_t kArena = (1024u * 1024u + 2u * 1024u) * sizeof(float) + 4096u;

} // namespace

static void BM_F32_Scalar(benchmark::State& st) {
    const auto M = static_cast<std::size_t>(st.range(0));
    const auto K = static_cast<std::size_t>(st.range(1));
    MemoryPool pool(kArena);
    auto W = Tensor<float>::create(pool, {M, K});
    auto x = Tensor<float>::create(pool, {K});
    auto y = Tensor<float>::create(pool, {M});
    fill_f32(W, 1); fill_f32(x, 2);
    for (auto _ : st) {
        C::kernels::matvec_f32_scalar(y.data(), W.data(), x.data(), M, K);
        benchmark::DoNotOptimize(y.data());
        benchmark::ClobberMemory();
    }
    st.SetItemsProcessed(st.iterations() * static_cast<std::int64_t>(M * K));
}

static void BM_F32_AVX2(benchmark::State& st) {
    const auto M = static_cast<std::size_t>(st.range(0));
    const auto K = static_cast<std::size_t>(st.range(1));
    MemoryPool pool(kArena);
    auto W = Tensor<float>::create(pool, {M, K});
    auto x = Tensor<float>::create(pool, {K});
    auto y = Tensor<float>::create(pool, {M});
    fill_f32(W, 1); fill_f32(x, 2);
    for (auto _ : st) {
        C::kernels::matvec_f32_avx2(y.data(), W.data(), x.data(), M, K);
        benchmark::DoNotOptimize(y.data());
        benchmark::ClobberMemory();
    }
    st.SetItemsProcessed(st.iterations() * static_cast<std::int64_t>(M * K));
}

static void BM_I8_Scalar(benchmark::State& st) {
    const auto M = static_cast<std::size_t>(st.range(0));
    const auto K = static_cast<std::size_t>(st.range(1));
    MemoryPool pool(kArena);
    auto W = Tensor<std::int8_t>::create(pool, {M, K});
    auto x = Tensor<std::int8_t>::create(pool, {K});
    auto y = Tensor<float>::create(pool, {M});
    fill_i8(W, 1); fill_i8(x, 2);
    for (auto _ : st) {
        C::kernels::matvec_i8_scalar(y.data(), W.data(), x.data(), M, K, 0.01f);
        benchmark::DoNotOptimize(y.data());
        benchmark::ClobberMemory();
    }
    st.SetItemsProcessed(st.iterations() * static_cast<std::int64_t>(M * K));
}

static void BM_I8_AVX2(benchmark::State& st) {
    const auto M = static_cast<std::size_t>(st.range(0));
    const auto K = static_cast<std::size_t>(st.range(1));
    MemoryPool pool(kArena);
    auto W = Tensor<std::int8_t>::create(pool, {M, K});
    auto x = Tensor<std::int8_t>::create(pool, {K});
    auto y = Tensor<float>::create(pool, {M});
    fill_i8(W, 1); fill_i8(x, 2);
    for (auto _ : st) {
        C::kernels::matvec_i8_avx2(y.data(), W.data(), x.data(), M, K, 0.01f);
        benchmark::DoNotOptimize(y.data());
        benchmark::ClobberMemory();
    }
    st.SetItemsProcessed(st.iterations() * static_cast<std::int64_t>(M * K));
}

#define SC_SIZES ->Args({256, 256})->Args({512, 512})->Args({1024, 1024})

BENCHMARK(BM_F32_Scalar) SC_SIZES;
BENCHMARK(BM_F32_AVX2)   SC_SIZES;
BENCHMARK(BM_I8_Scalar)  SC_SIZES;
BENCHMARK(BM_I8_AVX2)    SC_SIZES;

BENCHMARK_MAIN();
