add_executable(strata_infer strata_infer.cpp)
target_link_libraries(strata_infer PRIVATE Strata::Core strata_flags)

add_executable(strata_int8_infer strata_int8_infer.cpp)
target_link_libraries(strata_int8_infer PRIVATE Strata::Core strata_flags)

# A long-lived stdin/stdout inference process used by the local replay demo.
# It loads weights once, then accepts one CSV feature vector per line and
# returns a JSON result per line.  This keeps the dashboard on the same custom
# C++ engine as the CLI instead of reimplementing inference in Python.
add_executable(strata_stream strata_stream.cpp)
target_link_libraries(strata_stream PRIVATE Strata::Core strata_flags)

if(STRATA_BUILD_TESTS)
  add_test(
    NAME strata_infer_smoke
    COMMAND strata_infer
            "${CMAKE_SOURCE_DIR}/models/mlp.onnx"
            "${CMAKE_SOURCE_DIR}/models/mlp_io.csv"
            --oracle --iterations 32 --verify-repeats 16 --profile
            --compare-backends --trace)
  add_test(
    NAME strata_int8_smoke
    COMMAND strata_int8_infer
            "${CMAKE_SOURCE_DIR}/models/mlp.onnx"
            "${CMAKE_SOURCE_DIR}/models/mlp_io.csv"
            --iterations 16 --verify-fp32)
endif()
