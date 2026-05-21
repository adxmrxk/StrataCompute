@echo off
REM ===========================================================================
REM  StrataCompute — MSVC build driver
REM
REM  This box has VS2019 *Build Tools* (no IDE) which CMake's "Visual Studio"
REM  generator cannot auto-discover via vswhere. We instead initialise the
REM  MSVC environment with vcvars64.bat and use the NMake Makefiles generator.
REM
REM  Usage:  scripts\build-msvc.bat [Release|Debug]
REM ===========================================================================
setlocal
set CFG=%1
if "%CFG%"=="" set CFG=Release

set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "CMAKE=C:\Users\AdamR\AppData\Roaming\Python\Python312\site-packages\cmake\data\bin\cmake.exe"
set "CTEST=C:\Users\AdamR\AppData\Roaming\Python\Python312\site-packages\cmake\data\bin\ctest.exe"

call "%VCVARS%" || exit /b 1

cd /d "%~dp0.."
"%CMAKE%" -S . -B build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=%CFG% || exit /b 1
"%CMAKE%" --build build || exit /b 1
"%CTEST%" --test-dir build --output-on-failure || exit /b 1

echo.
echo BUILD+TESTS OK (%CFG%)
endlocal
