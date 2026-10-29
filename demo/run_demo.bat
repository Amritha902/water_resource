@echo off
rem Double-click to start the live demo.
cd /d "%~dp0\.."
python demo\run_demo.py %*
pause
