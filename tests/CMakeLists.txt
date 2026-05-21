# Offline test suite (Phases 1, 2, 4) — dependency-free, no network/ORT.
add_executable(strata_tests
  test_main.cpp
  test_memory_pool.cpp
  test_tensor.cpp
  test_compute.cpp
  test_engine.cpp)

target_include_directories(strata_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(strata_tests PRIVATE Strata::Core strata_flags)
# Lets the Phase 4 test locate models/ regardless of run cwd.
target_compile_definitions(strata_tests PRIVATE
  STRATA_MODELS_DIR="${CMAKE_SOURCE_DIR}/models")

add_test(NAME strata_tests COMMAND strata_tests)
