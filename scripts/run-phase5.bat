@echo off
REM ===========================================================================
REM  StrataCompute — Phase 5 driver
REM  Builds the C++ pipeline benchmark (ORT vs custom), runs it, then runs the
REM  PyTorch baseline + report generator. Produces docs/benchmarks.md.
REM
REM  Usage:  scripts\run-phase5.bat
REM ===========================================================================
setlocal
set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "CMAKE=C:\Users\AdamR\AppData\Roaming\Python\Python312\site-packages\cmake\data\bin\cmake.exe"

call "%VCVARS%" || exit /b 1
cd /d "%~dp0.."

"%CMAKE%" -S . -B build-onnx -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release ^
  -DSTRATA_BUILD_ONNX=ON -DSTRATA_BUILD_TESTS=ON || exit /b 1
"%CMAKE%" --build build-onnx --target bench_pipeline || exit /b 1

build-onnx\bench_pipeline.exe models results\latency_cpp.csv || exit /b 1
python scripts\run_phase5.py || exit /b 1

echo.
echo PHASE 5 OK - see docs\benchmarks.md
endlocal
