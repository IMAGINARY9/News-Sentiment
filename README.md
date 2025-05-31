# News Sentiment Analysis

This project implements sentiment analysis for news articles and tweets, supporting both transformer-based and LSTM-based models, with robust preprocessing and long document handling.

## Overview

- **Task:** Classify sentiment of news articles and tweets (positive, negative, neutral)
- **Challenges:** Long documents, domain-specific language (financial, political news), social media text
- **Approach:** Fine-tuned transformer models, LSTM models, hierarchical processing for long texts, and tweet-specific preprocessing

## Project Structure

```
News-Sentiment/
├── data/                    # News and financial datasets
│   └── your_dataset_name/   # Dataset specified folder 
│       └── raw/             # Dataset raw
│       └── processes/       # Dataset after data preparation
│       └── splits/          # Dataset splits for training
├── src/                     # Source code (models, preprocessing, training, evaluation)
├── models/                  # Trained models and checkpoints
├── notebooks/               # Jupyter notebooks for experiments and EDA
├── configs/                 # Model and training configuration files
├── scripts/                 # Training and evaluation scripts
├── logs/                    # Training logs
├── reports/                 # Analysis reports and results
├── tests/                   # Unit tests
└── visualizations/          # Visualization scripts and outputs
```

## Key Features

- **Long Document Handling:** Hierarchical models, chunking strategies for BERT/transformers
- **Domain Adaptation:** Financial news (FinBERT), political news, and generic news
- **Model Architectures:** 
  - Transformer-based (BERT, RoBERTa, DeBERTa, FinBERT)
  - LSTM-based (bidirectional LSTM for tweets/news)
- **Preprocessing:** 
  - NewsPreprocessor for articles (cleaning, chunking, tokenization)
  - TweetPreprocessor for tweets (stopword removal, stemming, vocab building)
- **Evaluation:** Domain-specific metrics, robust validation, and reporting

## Notebooks

- `notebooks/exploration.ipynb`: Data exploration and visualization
- `notebooks/preprocessed_exploration.ipynb`: Preprocessing and feature analysis
- Reference: See also `references/twitter-sentiment-analysis-lstm.ipynb` for LSTM pipeline

## Setup

### Automated Setup

**Windows (PowerShell):**
```powershell
.\setup.ps1
.\activate.bat
```

**Unix/Linux/macOS:**
```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

### Manual Setup

1. Create virtual environment: `python -m venv venv`
2. Activate: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Unix/Linux)
3. Install dependencies: `pip install -r requirements.txt`

## Usage

### Training a Model

**Transformer-based (BERT, RoBERTa, FinBERT, etc):**
```powershell
python scripts/train.py --config configs/base_config.yaml
```

**LSTM-based (for tweets/news):**
```powershell
python scripts/train.py --config configs/lstm_news.yaml
```
- The config file should specify `type: lstm` in the `model` section.

### Evaluating a Model

```powershell
python scripts/evaluate.py --model-path models/best_model.pt --config configs/base_config.yaml
```
or
```powershell
python scripts/evaluate.py --model-path models/best_model.pt --config configs/lstm_news.yaml
```

### Predicting Sentiment

You can predict sentiment for new text or a batch of texts using the trained model:

**Single text prediction:**
```powershell
python scripts/predict.py --model-path models/best_model.pt --config configs/base_config.yaml --text "Your news text here"
```

**Batch prediction from file:**
```powershell
python scripts/predict.py --model-path models/best_model.pt --config configs/base_config.yaml --file data/your_input_file.txt
```
- The input file should contain one document per line.
- The script prints predictions and confidence for each input.

### Data Preparation

Open and run the notebook:

```markdown
notebooks/data_preparation.ipynb
```

Follow the step-by-step instructions in the notebook to prepare, clean, and split your data for training and evaluation.

## Datasets

- Financial PhraseBank (financial news sentiment)
- Custom news datasets (see `data/`)
- SemEval news-related tasks

## Models

- Fine-tuned BERT/RoBERTa/DeBERTa for news sentiment
- FinBERT for financial news
- Bidirectional LSTM for tweets/news (see `src/models.py` and `src/preprocessing.py`)

## Extending

- Add new model configs in `configs/`
- Add new preprocessing logic in `src/preprocessing.py`
- Add new model architectures in `src/models.py`
- Use or adapt notebooks in `notebooks/` and `references/` for experiments
