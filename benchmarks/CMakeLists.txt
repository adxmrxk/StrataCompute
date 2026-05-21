# Phase 2 micro-benchmarks. Google Benchmark is fetched here (needs network
# at configure time) and is OFF by default so Phase 1 builds offline.
# Enable with -DSTRATA_BUILD_BENCHMARKS=ON.
include(FetchContent)

set(BENCHMARK_ENABLE_TESTING     OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL     OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_WERROR      OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG        v1.8.3)
FetchContent_MakeAvailable(googlebenchmark)

# Benchmark TUs include third-party <benchmark/benchmark.h>; do NOT inherit
# strata_flags' /WX here (only first-party code is held to zero-warning).
# We still want the optimisation/ISA baseline for a meaningful measurement.
add_library(strata_bench_flags INTERFACE)
if(MSVC)
  target_compile_options(strata_bench_flags INTERFACE
    /EHsc $<$<CONFIG:Release>:/O2> $<$<CONFIG:Release>:/arch:AVX2>)
else()
  target_compile_options(strata_bench_flags INTERFACE
    $<$<CONFIG:Release>:-O3> $<$<CONFIG:Release>:-march=native>)
endif()

foreach(bench bench_matvec bench_memory_pool)
  add_executable(${bench} ${bench}.cpp)
  target_link_libraries(${bench}
    PRIVATE Strata::Core strata_bench_flags benchmark::benchmark)
endforeach()
