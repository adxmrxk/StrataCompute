@echo off
REM ===========================================================================
REM StrataMotion — one-command local, no-hardware dashboard demo.
REM Downloads only UCI HAR (58.2 MB, CC BY 4.0) on first use.  The dashboard
REM binds to 127.0.0.1 and streams replay data into the custom C++ engine.
REM ===========================================================================
setlocal
cd /d "%~dp0.."

python scripts\get_uci_har.py || exit /b 1
python scripts\train_har_demo.py || exit /b 1
call scripts\build-msvc.bat Release || exit /b 1

echo.
echo Open http://127.0.0.1:8080 in a browser. Press Ctrl+C here to stop it.
python demo\server.py
endlocal
