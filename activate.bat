@echo off
REM News Sentiment Analysis - Activation Script
REM This script activates the virtual environment

if not exist venv (
    echo Error: Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)

echo Activating News Sentiment Analysis environment...
call venv\Scripts\activate.bat

echo Environment activated successfully!
echo Current project: News Sentiment Analysis
echo To deactivate, type: deactivate
