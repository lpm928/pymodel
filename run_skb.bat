@echo off
echo Starting Antigravity SKB Data Module...
cd /d "%~dp0"

set PYTHON_EXE="D:\AI\ANACONDA\python.exe"

:: Try to install dependencies if not present (quietly)
echo Checking dependencies...
%PYTHON_EXE% -m pip install streamlit pandas scikit-learn >nul 2>&1

:: Run the application using python -m streamlit to avoid PATH issues
echo Launching Streamlit App...
%PYTHON_EXE% -m streamlit run app.py

if %errorlevel% neq 0 (
    echo.
    echo Error: Failed to launch the application.
    pause
)
