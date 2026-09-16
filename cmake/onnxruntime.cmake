# =============================================================================
#  ONNX Runtime (prebuilt) — dual-execution benchmark baseline.
#
#  The spec uses the ONNX Runtime C++ API as a *baseline*, not as part of the
#  StrataCompute hot path. Building ORT from source needs Python/protobuf and
#  is enormous; we instead fetch Microsoft's official prebuilt CPU release and
#  expose it as an imported target. Network is required at configure time, so
#  this is gated behind STRATA_BUILD_ONNX (OFF by default) — Phases 1/2 stay
#  buildable offline.
#
#  Cross-platform: Windows uses the win-x64 .zip (.lib import + .dll runtime);
#  Linux (the Docker target) uses the linux-x64 .tgz (libonnxruntime.so). This
#  is what lets the container build the project with the spec's *original*
#  GCC toolchain.
# =============================================================================
set(STRATA_ORT_VERSION "1.17.3" CACHE STRING "ONNX Runtime version")

if(WIN32)
  set(_ort_plat "win-x64")
  set(_ort_ext  "zip")
elseif(UNIX AND NOT APPLE)
  set(_ort_plat "linux-x64")
  set(_ort_ext  "tgz")
else()
  message(FATAL_ERROR "ONNX Runtime fetch: unsupported platform")
endif()

set(_ort_name "onnxruntime-${_ort_plat}-${STRATA_ORT_VERSION}")
set(_ort_url
    "https://github.com/microsoft/onnxruntime/releases/download/v${STRATA_ORT_VERSION}/${_ort_name}.${_ort_ext}")
set(_ort_dir  "${CMAKE_BINARY_DIR}/_ort")
set(_ort_root "${_ort_dir}/${_ort_name}")

if(NOT EXISTS "${_ort_root}/include/onnxruntime_cxx_api.h")
  set(_ort_arc "${_ort_dir}/${_ort_name}.${_ort_ext}")
  message(STATUS "Downloading ONNX Runtime ${STRATA_ORT_VERSION} (${_ort_plat}) ...")
  file(DOWNLOAD "${_ort_url}" "${_ort_arc}"
       STATUS _dl TLS_VERIFY ON SHOW_PROGRESS)
  list(GET _dl 0 _dl_code)
  if(NOT _dl_code EQUAL 0)
    # Some locked-down Windows hosts have a broken Schannel credential store
    # while Python's verified TLS stack remains usable. Keep CMake as the
    # primary downloader, but use that standard-library fallback so a clean
    # build does not depend on a pre-populated build directory.
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
      message(WARNING "CMake download failed (${_dl}); retrying with Python TLS")
      execute_process(
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_LIST_DIR}/download_with_python.py"
                "${_ort_url}" "${_ort_arc}"
        RESULT_VARIABLE _py_dl_code
        OUTPUT_VARIABLE _py_dl_out
        ERROR_VARIABLE _py_dl_err)
      if(NOT _py_dl_code EQUAL 0)
        message(FATAL_ERROR
          "ONNX Runtime download failed through CMake (${_dl}) and Python "
          "(${_py_dl_err})")
      endif()
    else()
      message(FATAL_ERROR "ONNX Runtime download failed: ${_dl}")
    endif()
  endif()
  file(ARCHIVE_EXTRACT INPUT "${_ort_arc}" DESTINATION "${_ort_dir}")
endif()

add_library(onnxruntime SHARED IMPORTED GLOBAL)
if(WIN32)
  set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION             "${_ort_root}/lib/onnxruntime.dll"
    IMPORTED_IMPLIB               "${_ort_root}/lib/onnxruntime.lib"
    INTERFACE_INCLUDE_DIRECTORIES "${_ort_root}/include")
  set(STRATA_ORT_RUNTIME "${_ort_root}/lib/onnxruntime.dll"
      CACHE INTERNAL "ORT runtime library")
else()
  set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION             "${_ort_root}/lib/libonnxruntime.so"
    INTERFACE_INCLUDE_DIRECTORIES "${_ort_root}/include")
  set(STRATA_ORT_RUNTIME "${_ort_root}/lib/libonnxruntime.so"
      CACHE INTERNAL "ORT runtime library")
endif()
set(STRATA_ORT_LIBDIR "${_ort_root}/lib" CACHE INTERNAL "ORT lib dir")

# Make an ORT-linked executable find the runtime: copy the DLL next to the
# exe on Windows; bake an RPATH to the .so dir on Linux.
function(strata_ort_runtime tgt)
  if(WIN32)
    add_custom_command(TARGET ${tgt} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              "${STRATA_ORT_RUNTIME}" "$<TARGET_FILE_DIR:${tgt}>")
  else()
    set_target_properties(${tgt} PROPERTIES
      BUILD_RPATH "${STRATA_ORT_LIBDIR}")
  endif()
endfunction()

message(STATUS "ONNX Runtime ready: ${_ort_root}")
