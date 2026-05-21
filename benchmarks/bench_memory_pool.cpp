// Phase 2 placeholder — arena bump-allocate vs. std::malloc on the hot path.
// Compiled only when -DSTRATA_BUILD_BENCHMARKS=ON (needs network for the
// Google Benchmark fetch). Real SIMD/GEMM benchmarks land in Phase 2.
#include "strata/memory_pool.hpp"
#include <benchmark/benchmark.h>

static void BM_ArenaBumpAllocate(benchmark::State& state) {
    Strata::MemoryPool pool(1 << 20);
    for (auto _ : state) {
        void* p = pool.try_allocate(256, 64);
        benchmark::DoNotOptimize(p);
        pool.reset();
    }
}
BENCHMARK(BM_ArenaBumpAllocate);

BENCHMARK_MAIN();
