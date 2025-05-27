@echo off
REM News Sentiment Analysis - Setup Script (Windows Batch)
REM This script sets up the virtual environment and installs dependencies

echo Setting up News Sentiment Analysis environment...

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo Error: Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
if exist requirements.txt (
    echo Installing dependencies from requirements.txt...
    pip install -r requirements.txt
) else (
    echo Warning: requirements.txt not found
)

REM Install project in development mode
if exist setup.py (
    echo Installing project in development mode...
    pip install -e .
)

REM Create necessary directories
for %%d in (data models logs reports) do (
    if not exist %%d (
        mkdir %%d
        echo Created directory: %%d
    )
)

echo Setup completed successfully!
echo Virtual environment is now active.
echo To activate in future sessions, run: venv\Scripts\activate.bat
pause
