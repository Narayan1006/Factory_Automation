@echo off
title Bosch Smart Factory - AI Digital Twin
color 0A
cls
echo ======================================================================
echo    BOSCH SMART FACTORY - AI DIGITAL TWIN (INTERACTIVE CONTROL CENTER)
echo ======================================================================
echo.
echo  Welcome! Starting your beginner-friendly factory dashboard...
echo.

cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo [1/2] Checking Python environment...
%PYTHON_EXE% -c "import streamlit" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Installing required packages (streamlit)...
    %PYTHON_EXE% -m pip install streamlit
)

echo [2/2] Launching interactive web interface on http://localhost:8501...
start "" "http://localhost:8501"
%PYTHON_EXE% -m streamlit run app.py --server.port 8501

pause
