# PowerShell script for setting up the News Sentiment Analysis project on Windows

param(
    [switch]$UseCPUOnly,
    [switch]$SkipGPUCheck,
    [switch]$Force
)

# Set up error handling
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"  # Makes downloads faster

# Print banner
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "  News Sentiment Analysis Project - Windows Setup      " -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan

# Check if Python is installed
try {
    $pythonVersion = python --version
    Write-Host "✓ Found $pythonVersion" -ForegroundColor Green
    
    # Check Python version
    $versionString = $pythonVersion -replace "Python ", ""
    $versionParts = $versionString.Split(".")
    $majorVersion = [int]$versionParts[0]
    $minorVersion = [int]$versionParts[1]
    
    if ($majorVersion -lt 3 -or ($majorVersion -eq 3 -and $minorVersion -lt 8)) {
        Write-Host "⚠️ Warning: Python 3.8+ is recommended (found $versionString)" -ForegroundColor Yellow
        
        if (-Not $Force) {
            $confirmation = Read-Host "Continue anyway? (y/n)"
            if ($confirmation -ne 'y') {
                exit 1
            }
        }
    }
} catch {
    Write-Host "✕ Python not found. Please install Python 3.8 or newer." -ForegroundColor Red
    Write-Host "You can download Python from: https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}

# Check if virtual environment exists, create if not
if (-Not (Test-Path -Path ".\venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    
    if (-Not (Test-Path -Path ".\venv")) {
        Write-Host "✕ Failed to create virtual environment." -ForegroundColor Red
        exit 1
    }
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "✓ Virtual environment exists" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
try {
    & .\venv\Scripts\Activate.ps1
} catch {
    Write-Host "✕ Failed to activate virtual environment due to PowerShell execution policy." -ForegroundColor Red
    Write-Host "Attempting to set execution policy for current user..." -ForegroundColor Yellow
    
    try {
        Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
        & .\venv\Scripts\Activate.ps1
    } catch {
        Write-Host "✕ Still unable to activate virtual environment." -ForegroundColor Red
        Write-Host "Please run the following command manually and try again:" -ForegroundColor Yellow
        Write-Host "    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor White
        exit 1
    }
}

# Check if activation was successful
if (-Not $env:VIRTUAL_ENV) {
    Write-Host "✕ Failed to activate virtual environment." -ForegroundColor Red
    Write-Host "You may need to run: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor Yellow
    exit 1
}
Write-Host "✓ Virtual environment activated" -ForegroundColor Green

# Install pip tools first for better dependency handling
Write-Host "Upgrading pip and installing key tools..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow

if ($UseCPUOnly) {
    # Explicitly disable GPU to avoid any detection attempts
    Write-Host "Installing for CPU-only operation..." -ForegroundColor Cyan
    $env:CUDA_VISIBLE_DEVICES = "-1"
    python -m pip install -r requirements.txt
} else {
    # Standard installation
    python -m pip install -r requirements.txt
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "✕ Failed to install dependencies." -ForegroundColor Red
    exit 1
}
Write-Host "✓ Dependencies installed" -ForegroundColor Green

# Install Jupyter kernel for the virtual environment
Write-Host "Installing Jupyter kernel for this environment..." -ForegroundColor Yellow
python -m ipykernel install --user --name=news-sentiment-env --display-name="Python (News Sentiment)"
Write-Host "✓ Jupyter kernel installed" -ForegroundColor Green

# Create necessary directories if they don't exist
$directories = @("data", "models", "logs", "outputs", "reports", "notebooks", "visualizations", "cache")
foreach ($dir in $directories) {
    if (-Not (Test-Path -Path ".\$dir")) {
        New-Item -Path ".\$dir" -ItemType Directory | Out-Null
        Write-Host "Created directory: $dir" -ForegroundColor Yellow
    } else {
        Write-Host "✓ Directory exists: $dir" -ForegroundColor Green
    }
}

# Create a .pth file in the site-packages directory to automatically add the project to Python path
$sitePackagesDir = Join-Path -Path ".\venv\Lib\site-packages" -ChildPath "news_sentiment.pth"
$projectPath = (Get-Item -Path ".").FullName
$projectPath | Out-File -FilePath $sitePackagesDir -Encoding ascii
Write-Host "✓ Created .pth file for automatic Python path configuration" -ForegroundColor Green

# Download NLTK data
Write-Host "Downloading NLTK data..." -ForegroundColor Yellow
$nltkDownloads = @("punkt", "stopwords", "wordnet", "vader_lexicon")
foreach ($corpus in $nltkDownloads) {
    python -c "import nltk; nltk.download('$corpus', quiet=True)"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Downloaded NLTK corpus: $corpus" -ForegroundColor Green
    } else {
        Write-Host "⚠️ Warning: Failed to download NLTK corpus: $corpus" -ForegroundColor Yellow
    }
}

# Download pre-trained models
Write-Host "Downloading pre-trained models..." -ForegroundColor Yellow
$models = @(
    @{name="BERT Base"; repo="bert-base-uncased"},
    @{name="FinBERT"; repo="ProsusAI/finbert"}
)

foreach ($model in $models) {
    Write-Host "Downloading $($model.name)..." -ForegroundColor Cyan
    python -c "from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained('$($model.repo)'); AutoTokenizer.from_pretrained('$($model.repo)')"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Downloaded $($model.name)" -ForegroundColor Green
    } else {
        Write-Host "⚠️ Warning: Failed to download $($model.name)" -ForegroundColor Yellow
    }
}

# Check for TensorFlow/PyTorch and GPU availability (skip if requested)
if (-Not $SkipGPUCheck) {
    Write-Host "Checking ML frameworks and GPU availability..." -ForegroundColor Yellow

    $gpuCheckScript = @"
import os
import sys
import platform

# Try to configure frameworks to show minimal logs but still show errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def check_frameworks():
    print(f"Python version: {platform.python_version()}")
    print(f"Platform: {platform.system()} {platform.release()}")
    
    # Check PyTorch
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"PyTorch CUDA available: {torch.cuda.device_count()} GPU(s)")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("PyTorch: No CUDA GPU detected")
    except ImportError:
        print("PyTorch not installed")
    except Exception as e:
        print(f"PyTorch error: {str(e)}")
    
    # Check TensorFlow
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"TensorFlow GPU(s) detected: {len(gpus)}")
            for gpu in gpus:
                print(f"  {gpu.name}")
        else:
            print("TensorFlow: No GPU detected")
    except ImportError:
        print("TensorFlow not installed")
    except Exception as e:
        print(f"TensorFlow error: {str(e)}")

if __name__ == "__main__":
    check_frameworks()
"@

    $gpuCheckScript | Out-File -FilePath ".\gpu_check.py" -Encoding utf8
    python .\gpu_check.py
    Remove-Item -Path ".\gpu_check.py"
}

# Create activate.bat for easier environment activation
$activateBatContent = @"
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
"@

$activateBatContent | Out-File -FilePath ".\activate.bat" -Encoding ascii
Write-Host "✓ Created activate.bat for easy environment activation" -ForegroundColor Green

# Write summary
Write-Host "`n=======================================================" -ForegroundColor Cyan
Write-Host " Setup Complete! Environment is ready!" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan

Write-Host "`nQuick Start Guide:" -ForegroundColor White
Write-Host "1. Activate the environment:" -ForegroundColor White
Write-Host "   - Using PowerShell: .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host "   - Using CMD:        .\activate.bat" -ForegroundColor Yellow

Write-Host "`n2. Prepare your data:" -ForegroundColor White
Write-Host "   python scripts\prepare_data.py" -ForegroundColor Yellow

Write-Host "`n3. Train a model:" -ForegroundColor White
Write-Host "   python scripts\train.py --config configs\base_config.yaml" -ForegroundColor Yellow

Write-Host "`n4. Evaluate a model:" -ForegroundColor White
Write-Host "   python scripts\evaluate.py --model models\best_model.pt" -ForegroundColor Yellow

Write-Host "`n5. For Jupyter notebook:" -ForegroundColor White
Write-Host "   jupyter notebook" -ForegroundColor Yellow

if ($UseCPUOnly) {
    Write-Host "`n6. Using CPU-only mode" -ForegroundColor White
    Write-Host "   You've configured the environment for CPU-only operation" -ForegroundColor Yellow
}

Write-Host "`nTroubleshooting:" -ForegroundColor White
Write-Host "If you encounter import errors, check the following:" -ForegroundColor Yellow
Write-Host "1. Make sure the virtual environment is activated" -ForegroundColor Yellow
Write-Host "2. Use the activate.bat script which sets up PYTHONPATH correctly" -ForegroundColor Yellow
Write-Host "3. Add the project directory to PYTHONPATH manually:" -ForegroundColor Yellow
Write-Host "   `$env:PYTHONPATH = '$((Get-Item -Path '.').FullName);`$env:PYTHONPATH'" -ForegroundColor White

Write-Host "`nFor help with specific commands, run them with --help flag" -ForegroundColor White
