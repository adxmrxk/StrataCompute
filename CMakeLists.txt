# =============================================================================
# StrataCompute — high-performance C++ inference engine
# Root build script.
#
# Phase 1 deliverable: Strata::MemoryPool + Strata::Tensor, built and verified
# with a modern toolchain. Build flags are abstracted per-compiler so the same
# tree builds with MSVC (this dev box: VS2019 Build Tools) *and* with
# GCC/Clang (-O3 -march=native -flto) on deployment hardware.
#
# Hardware note: the spec targets AVX-512; the current dev CPU (i5-12400,
# Alder Lake) has AVX-512 fused off. SIMD work (Phase 2) therefore targets
# AVX2/FMA/VNNI with a runtime-dispatch seam reserved for an AVX-512 kernel.
# =============================================================================
cmake_minimum_required(VERSION 3.20)

project(StrataCompute
        VERSION 0.1.0
        DESCRIPTION "Bare-metal C++ ML inference engine"
        LANGUAGES CXX)

# ---- Standard --------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

# ---- Options ---------------------------------------------------------------
option(STRATA_BUILD_TESTS      "Build the test suite"                    ON)
option(STRATA_BUILD_BENCHMARKS "Build Google Benchmark micro-benchmarks" OFF)
option(STRATA_BUILD_ONNX       "Build ONNX Runtime baseline (downloads ORT)" OFF)
option(STRATA_BUILD_TOOLS      "Build command-line inference tools"     ON)
option(STRATA_WERROR           "Treat warnings as errors"                ON)

# ---- Compiler flag abstraction --------------------------------------------
# A single INTERFACE target carries the optimisation / warning policy so every
# component links the same contract regardless of toolchain.
add_library(strata_flags INTERFACE)

if(MSVC)
  target_compile_options(strata_flags INTERFACE
    /permissive-            # strict standard conformance
    /Zc:__cplusplus         # report the real __cplusplus value
    /EHsc
    /W4
    $<$<BOOL:${STRATA_WERROR}>:/WX>
    $<$<CONFIG:Release>:/O2>
    $<$<CONFIG:Release>:/GL>            # whole-program opt (LTO equivalent)
    $<$<CONFIG:Release>:/arch:AVX2>)    # AVX2/FMA — AVX-512 absent on dev CPU
  target_link_options(strata_flags INTERFACE
    $<$<CONFIG:Release>:/LTCG>)
else()
  # GCC / Clang (deployment hardware, e.g. AVX-512 Xeon)
  target_compile_options(strata_flags INTERFACE
    -Wall -Wextra -Wpedantic
    $<$<BOOL:${STRATA_WERROR}>:-Werror>
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Release>:-march=native>
    $<$<CONFIG:Release>:-flto>)
  target_link_options(strata_flags INTERFACE
    $<$<CONFIG:Release>:-flto>)
endif()

# ---- Core library ----------------------------------------------------------
add_library(strata_core STATIC
  src/memory_pool.cpp
  src/compute/cpu_features.cpp
  src/compute/matvec_scalar.cpp
  src/compute/matvec_avx2.cpp
  src/compute/matvec_dispatch.cpp
  src/compute/elementwise.cpp
  src/onnx/onnx_graph.cpp
  src/engine/forward_engine.cpp)
target_include_directories(strata_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
# PRIVATE: the zero-warning/opt policy applies to strata_core's own TUs, not
# forced onto consumers (e.g. bench_pipeline pulls in third-party ORT headers
# and must not inherit /WX). First-party consumers link strata_flags directly.
target_link_libraries(strata_core PRIVATE strata_flags)
add_library(Strata::Core ALIAS strata_core)

# ---- Tests -----------------------------------------------------------------
if(STRATA_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# ---- Tools -----------------------------------------------------------------
# The inference CLI is deliberately linked only to Strata::Core: it proves the
# custom engine is useful without requiring ONNX Runtime at deployment time.
# This follows enable_testing() so its smoke test is registered with CTest.
if(STRATA_BUILD_TOOLS)
  add_subdirectory(tools)
endif()

# ---- Benchmarks (Phase 2; fetched only when explicitly enabled) -----------
if(STRATA_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

# ---- ONNX Runtime baseline (Phase 3; downloads ORT when enabled) ----------
if(STRATA_BUILD_ONNX)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/onnxruntime.cmake)

  # ORT's <onnxruntime_cxx_api.h> is third-party; hold our code to /W3 (no
  # /WX) like the benchmark TUs — first-party StrataCompute stays zero-warning.
  add_library(strata_onnx_flags INTERFACE)
  if(MSVC)
    target_compile_options(strata_onnx_flags INTERFACE
      /EHsc /W3 $<$<CONFIG:Release>:/O2> $<$<CONFIG:Release>:/arch:AVX2>)
  else()
    target_compile_options(strata_onnx_flags INTERFACE
      $<$<CONFIG:Release>:-O3> $<$<CONFIG:Release>:-march=native>)
  endif()

  add_library(strata_onnx STATIC src/onnx/onnx_model.cpp)
  target_include_directories(strata_onnx PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(strata_onnx
    PUBLIC onnxruntime PRIVATE strata_onnx_flags)
  add_library(Strata::Onnx ALIAS strata_onnx)

  if(STRATA_BUILD_TESTS)
    add_executable(strata_onnx_test tests/test_onnx.cpp)
    target_include_directories(strata_onnx_test
      PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    target_link_libraries(strata_onnx_test
      PRIVATE Strata::Onnx strata_onnx_flags)
    add_test(NAME strata_onnx_test
             COMMAND strata_onnx_test "${CMAKE_CURRENT_SOURCE_DIR}/models")
    strata_ort_runtime(strata_onnx_test)   # Win: copy dll / Linux: rpath .so
  endif()

  # Phase 5: three-way latency benchmark (ORT vs custom). Hand-rolled timing
  # (not Google Benchmark) for explicit P95/P99 + CSV export.
  add_executable(bench_pipeline benchmarks/bench_pipeline.cpp)
  target_link_libraries(bench_pipeline
    PRIVATE Strata::Onnx Strata::Core strata_onnx_flags)
  strata_ort_runtime(bench_pipeline)
endif()

message(STATUS "StrataCompute ${PROJECT_VERSION} | "
        "compiler=${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} | "
        "tests=${STRATA_BUILD_TESTS} benchmarks=${STRATA_BUILD_BENCHMARKS} "
        "onnx=${STRATA_BUILD_ONNX}")
