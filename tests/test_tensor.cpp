// Phase 1 — Strata::Tensor behavioural tests.
#include "strata/tensor.hpp"
#include "strata/memory_pool.hpp"
#include "test_harness.hpp"

using Strata::MemoryPool;
using Strata::Tensor;
using Strata::Locality;

STRATA_TEST(tensor_shape_strides_and_count) {
    MemoryPool pool(1 << 16);
    auto t = Tensor<float>::create(pool, {2, 3, 4});

    STRATA_CHECK_EQ(t.rank(), 3u);
    STRATA_CHECK_EQ(t.shape(0), 2u);
    STRATA_CHECK_EQ(t.shape(1), 3u);
    STRATA_CHECK_EQ(t.shape(2), 4u);
    // Row-major strides: innermost == 1.
    STRATA_CHECK_EQ(t.stride(2), 1u);
    STRATA_CHECK_EQ(t.stride(1), 4u);
    STRATA_CHECK_EQ(t.stride(0), 12u);
    STRATA_CHECK_EQ(t.size(), 24u);
    STRATA_CHECK_EQ(t.nbytes(), 24u * sizeof(float));
}

STRATA_TEST(tensor_is_64byte_cache_aligned) {
    MemoryPool pool(1 << 16);
    auto a = Tensor<float>::create(pool, {7});      // odd size
    auto b = Tensor<double>::create(pool, {5, 5});  // next carve
    STRATA_CHECK(a.is_cache_aligned());
    STRATA_CHECK(b.is_cache_aligned());             // pool re-aligns each carve
}

STRATA_TEST(tensor_indexing_round_trip) {
    MemoryPool pool(1 << 16);
    auto t = Tensor<int>::create(pool, {3, 4});

    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            t(i, j) = static_cast<int>(i * 10 + j);

    STRATA_CHECK_EQ(t(0, 0), 0);
    STRATA_CHECK_EQ(t(2, 3), 23);
    STRATA_CHECK_EQ(t.index(2, 3), 11u);            // 2*4 + 3
    // Flat layout matches row-major expectation.
    STRATA_CHECK_EQ(t.flat()[11], 23);
}

STRATA_TEST(tensor_row_span_view) {
    MemoryPool pool(1 << 16);
    auto t = Tensor<float>::create(pool, {3, 5});
    for (std::size_t j = 0; j < 5; ++j) t(1, j) = static_cast<float>(j) + 0.5f;

    std::span<float> r = t.row(1);
    STRATA_CHECK_EQ(r.size(), 5u);
    STRATA_CHECK_EQ(r[0], 0.5f);
    STRATA_CHECK_EQ(r[4], 4.5f);
}

STRATA_TEST(tensor_wrap_non_owning) {
    alignas(Strata::kCacheLineBytes) float buf[8] = {};
    auto t = Tensor<float>::wrap(buf, {2, 4});
    t(1, 3) = 9.0f;
    STRATA_CHECK_EQ(buf[7], 9.0f);                   // writes through, no copy
    STRATA_CHECK_EQ(t.data(), buf);
}

STRATA_TEST(tensor_prefetch_is_safe_noop_on_valid_addr) {
    MemoryPool pool(1 << 16);
    auto t = Tensor<float>::create(pool, {4, 16});
    // Must not fault for any locality hint; purely a perf primitive.
    t.prefetch_row(0, Locality::L1);
    t.prefetch_row(3, Locality::L2);
    Tensor<float>::prefetch(t.data(), Locality::NonTemporal);
    STRATA_CHECK(true);
}
