# News Sentiment Analysis

This project implements sentiment analysis for news articles, focusing on handling long documents and domain-specific language.

## Overview

- **Task**: Classify sentiment of news articles (positive, negative, neutral)
- **Challenges**: Long documents, domain-specific language (financial, political news)
- **Approach**: Fine-tuned transformer models with hierarchical processing for long texts

## Project Structure

```
news-sentiment/
├── data/                    # News-specific datasets
├── src/                     # Source code
├── models/                  # Trained models and checkpoints
├── notebooks/               # Jupyter notebooks for experiments
├── configs/                 # Configuration files
├── scripts/                 # Training and evaluation scripts
├── references/              # Reference notebooks and papers
├── logs/                    # Training logs
├── reports/                 # Analysis reports and results
└── tests/                   # Unit tests
```

## Key Features

1. **Long Document Handling**: Hierarchical models, chunking strategies
2. **Domain Adaptation**: Financial news, political news specialization
3. **Model Architectures**: BERT, RoBERTa, DeBERTa with custom heads
4. **Evaluation**: Domain-specific metrics and analysis

## Setup

### Automated Setup (Recommended)

Use the provided setup scripts to create a virtual environment and install all dependencies:

**Windows:**
```bash
# Run the setup script
.\setup.bat

# Activate the environment
.\activate.bat
```

**PowerShell:**
```powershell
# Run the setup script
.\setup.ps1

# Activate the environment
.\activate.bat
```

**Unix/Linux/macOS:**
```bash
# Make script executable and run
chmod +x setup.sh
./setup.sh

# Activate the environment
source venv/bin/activate
```

### Manual Setup

If you prefer manual setup:
1. Create virtual environment: `python -m venv venv`
2. Activate environment: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Unix/Linux)
3. Install dependencies: `pip install -r requirements.txt`

## Quick Start

1. **Setup environment**: Use `setup.bat` (Windows) or `setup.sh` (Unix/Linux)
2. **Activate environment**: Run `activate.bat` or `source venv/bin/activate`
3. **Prepare data**: `python scripts/prepare_data.py`
4. **Train model**: `python scripts/train.py --config configs/bert_news.yaml`
5. **Evaluate**: `python scripts/evaluate.py --model-path models/best_model.pt`

## Datasets

- Financial PhraseBank (financial news sentiment)
- Custom news datasets
- SemEval news-related tasks

## Models

- Fine-tuned BERT/RoBERTa for news sentiment
- Hierarchical attention networks for long documents
- Domain-specific models (FinBERT for financial news)
