@echo off
cd /d "%~dp0.."
echo Starting Raw Material Monitor...
echo Press Ctrl+C to stop.
call .venv\Scripts\python.exe scripts\raw_material_monitor.py
if errorlevel 1 (
    echo Monitor crashed or failed to start.
    pause
)
