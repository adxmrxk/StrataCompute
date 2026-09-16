@echo off
REM StrataCompute MSVC build driver.
REM Uses the VS2019 Build Tools vcvars64 environment and NMake generator.
REM Usage: scripts\build-msvc.bat Release or scripts\build-msvc.bat Debug.
setlocal
set CFG=%1
if "%CFG%"=="" set CFG=Release

set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
call "%VCVARS%" || exit /b 1

cd /d "%~dp0.."
cmake -S . -B build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=%CFG% || exit /b 1
cmake --build build || exit /b 1
ctest --test-dir build --output-on-failure || exit /b 1

echo.
echo BUILD+TESTS OK (%CFG%)
endlocal
