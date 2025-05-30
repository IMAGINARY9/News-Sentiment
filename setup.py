"""
Setup script for news sentiment analysis project.

This script installs dependencies, downloads models, and prepares the environment.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description, use_venv=True):
    """Run a command and handle errors, optionally in venv."""
    print(f"\n{description}...")
    venv_python = os.path.join("venv", "Scripts", "python.exe")
    if use_venv and os.path.exists(venv_python):
        if command.startswith("python "):
            command = command.replace("python", f'"{venv_python}"', 1)
        elif command.startswith("pip "):
            command = command.replace("pip", f'"{venv_python}" -m pip', 1)
    try:
        result = subprocess.run(command, check=True, shell=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def ensure_venv():
    """Ensure the virtual environment exists, create if not."""
    venv_path = Path("venv")
    if not venv_path.exists():
        print("Creating virtual environment...")
        result = subprocess.run(f"python -m venv venv", shell=True)
        if result.returncode != 0:
            print("✗ Failed to create virtual environment")
            sys.exit(1)
        print("✓ Virtual environment created.")
    else:
        print("✓ Virtual environment already exists.")

def install_dependencies():
    """Install Python dependencies in venv."""
    venv_python = os.path.join("venv", "Scripts", "python.exe")
    # Upgrade pip, setuptools, wheel first
    run_command(f'"{venv_python}" -m pip install --upgrade pip setuptools wheel', "Upgrading pip and build tools", use_venv=False)
    return run_command(f'"{venv_python}" -m pip install -r requirements.txt', "Installing dependencies", use_venv=False)

def download_models():
    """Download pre-trained models using venv python."""
    venv_python = os.path.join("venv", "Scripts", "python.exe")
    commands = [
        f'"{venv_python}" -c "from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained(\'bert-base-uncased\'); AutoTokenizer.from_pretrained(\'bert-base-uncased\')"',
        f'"{venv_python}" -c "from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained(\'ProsusAI/finbert\'); AutoTokenizer.from_pretrained(\'ProsusAI/finbert\')"',
    ]
    success = True
    for cmd in commands:
        success &= run_command(cmd, "Downloading pre-trained models", use_venv=False)
    return success

def setup_directories():
    """Create necessary directories."""
    directories = [
        "data", "models", "logs", "outputs", "reports", 
        "notebooks", "visualizations", "cache"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    return True

def download_nltk_data():
    """Download required NLTK data using venv python."""
    venv_python = os.path.join("venv", "Scripts", "python.exe")
    commands = [
        f'"{venv_python}" -c "import nltk; nltk.download(\'punkt\', quiet=True)"',
        f'"{venv_python}" -c "import nltk; nltk.download(\'stopwords\', quiet=True)"',
        f'"{venv_python}" -c "import nltk; nltk.download(\'wordnet\', quiet=True)"',
    ]
    success = True
    for cmd in commands:
        success &= run_command(cmd, "Downloading NLTK data", use_venv=False)
    return success

def main():
    print("Setting up News Sentiment Analysis Project")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("✗ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✓ Using Python {sys.version}")
    
    # Ensure virtual environment is set up
    ensure_venv()
    
    # Setup steps
    steps = [
        ("Setting up directories", setup_directories),
        ("Installing dependencies", install_dependencies),
        ("Downloading NLTK data", download_nltk_data),
        ("Downloading pre-trained models", download_models),
    ]
    
    success_count = 0
    for description, func in steps:
        if func():
            success_count += 1
        else:
            print(f"⚠ Warning: {description} failed, but continuing...")
    
    print(f"\nSetup completed: {success_count}/{len(steps)} steps successful")
    
    if success_count == len(steps):
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Prepare your data: python scripts/prepare_data.py")
        print("2. Train a model: python scripts/train.py --config configs/base_config.yaml")
        print("3. Evaluate: python scripts/evaluate.py --model models/best_model.pt")
    else:
        print("\n⚠ Setup completed with some warnings. Check the output above.")

if __name__ == "__main__":
    main()
