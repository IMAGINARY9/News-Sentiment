@echo off
echo Activating News Sentiment Analysis environment...
call %~dp0venv\Scripts\activate.bat
echo.
echo Environment ready! You can now run:
echo   - python scripts\prepare_data.py       (Prepare your data)
echo   - python scripts\train.py              (Train a model)
echo   - python scripts\evaluate.py           (Evaluate a model)
echo   - jupyter notebook                     (Start Jupyter notebook server)
echo.
set PYTHONPATH=%~dp0;%PYTHONPATH%
