// Phase 1 — Strata::MemoryPool behavioural tests.
#include "strata/memory_pool.hpp"
#include "test_harness.hpp"

#include <cstdint>
#include <new>

using Strata::MemoryPool;
using Strata::kCacheLineBytes;

namespace {
std::uintptr_t addr(const void* p) {
    return reinterpret_cast<std::uintptr_t>(p);
}
} // namespace

STRATA_TEST(pool_base_is_cache_aligned) {
    MemoryPool pool(4096);
    STRATA_CHECK(pool.data() != nullptr);
    STRATA_CHECK_EQ(addr(pool.data()) % kCacheLineBytes, 0u);
    STRATA_CHECK_EQ(pool.capacity(), 4096u);
    STRATA_CHECK_EQ(pool.used(), 0u);
}

STRATA_TEST(allocate_bumps_and_aligns) {
    MemoryPool pool(1024);

    void* a = pool.allocate(10, 8);
    STRATA_CHECK_EQ(addr(a) % 8, 0u);
    STRATA_CHECK_EQ(pool.used(), 10u);          // 0 -> 10

    void* b = pool.allocate(4, 64);
    STRATA_CHECK_EQ(addr(b) % 64, 0u);          // cursor padded 10 -> 64
    STRATA_CHECK_EQ(pool.used(), 68u);          // 64 + 4
    STRATA_CHECK(addr(b) > addr(a));
}

STRATA_TEST(try_allocate_returns_null_on_exhaustion_no_throw) {
    MemoryPool pool(128);
    void* a = pool.try_allocate(100, 64);
    STRATA_CHECK(a != nullptr);
    void* b = pool.try_allocate(100, 64);       // would overflow
    STRATA_CHECK(b == nullptr);                  // no exception, no UB
    STRATA_CHECK(pool.used() <= pool.capacity());
}

STRATA_TEST(allocate_throws_bad_alloc_on_exhaustion) {
    MemoryPool pool(64);
    STRATA_CHECK_THROWS(pool.allocate(65, 1), std::bad_alloc);
}

STRATA_TEST(reset_recycles_and_keeps_high_water) {
    MemoryPool pool(512);
    STRATA_CHECK(pool.allocate(300, 64) != nullptr);
    const std::size_t hw = pool.high_water_mark();
    STRATA_CHECK(hw >= 300u);

    pool.reset();
    STRATA_CHECK_EQ(pool.used(), 0u);
    STRATA_CHECK_EQ(pool.high_water_mark(), hw); // retained for sizing

    void* p = pool.allocate(16, 64);             // reuses the front of arena
    STRATA_CHECK_EQ(addr(p) % 64, 0u);
}

STRATA_TEST(reset_zero_fill_scrubs_used_region) {
    MemoryPool pool(256);
    auto* p = static_cast<unsigned char*>(pool.allocate(64, 64));
    for (int i = 0; i < 64; ++i) p[i] = 0xAB;
    pool.reset(/*zero_fill=*/true);
    auto* q = static_cast<unsigned char*>(pool.allocate(64, 64));
    STRATA_CHECK(q == p);
    bool all_zero = true;
    for (int i = 0; i < 64; ++i) all_zero &= (q[i] == 0);
    STRATA_CHECK(all_zero);
}

STRATA_TEST(move_transfers_ownership) {
    MemoryPool a(256);
    STRATA_CHECK(a.allocate(32, 8) != nullptr);
    const void* base = a.data();

    MemoryPool b(std::move(a));
    STRATA_CHECK_EQ(b.data(), base);
    STRATA_CHECK_EQ(b.used(), 32u);
    STRATA_CHECK(a.data() == nullptr);           // moved-from is inert
}
