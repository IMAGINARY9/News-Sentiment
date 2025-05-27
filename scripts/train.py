"""
Training script for news sentiment analysis models.

Usage:
    python train.py --config configs/base_config.yaml
    python train.py --config configs/financial_config.yaml --model finbert
"""

import argparse
import yaml
import torch
import wandb
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from preprocessing import NewsPreprocessor
from models import create_model, NewsDataset
from training import NewsTrainer
from evaluation import NewsEvaluator
from transformers import AutoTokenizer

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Train news sentiment analysis model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--model', type=str, default=None, help='Model name override')
    parser.add_argument('--data', type=str, default=None, help='Data directory override')
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--debug', action='store_true', help='Debug mode with smaller dataset')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with CLI args
    if args.model:
        config['model']['name'] = args.model
    if args.data:
        config['paths']['data_dir'] = args.data
      # Initialize wandb if requested
    if args.wandb:
        wandb.init(
            project="news-sentiment",
            config=config,
            name=f"news_sentiment_{config['model']['name'].replace('/', '_')}"
        )
    
    print(f"Training news sentiment model with config: {args.config}")
    print(f"Model: {config['model']['name']}")
    print(f"Data directory: {config['paths']['data_dir']}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    
    # Initialize components
    preprocessor = NewsPreprocessor(
        tokenizer_name=config['model']['name'],
        max_length=config['model']['max_length']
    )
    
    model = create_model(config['model'])
    
    trainer = NewsTrainer(
        model=model,
        config=config['training']
    )
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    data_path = Path(config['paths']['data_dir'])
    
    # For financial news, load the financial phrase bank data
    if 'financial' in str(data_path):
        # Load financial sentiment data
        import pandas as pd
        
        # Load different agreement levels
        all_agree_path = data_path / "FinancialPhraseBank" / "Sentences_AllAgree.txt"
        if all_agree_path.exists():
            with open(all_agree_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            texts = []
            labels = []
            label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
            
            for line in lines:
                line = line.strip()
                if '@' in line:
                    # Split text and sentiment
                    parts = line.split('@')
                    if len(parts) == 2:
                        text = parts[0].strip()
                        sentiment = parts[1].strip().lower()
                        if sentiment in label_map:
                            texts.append(text)
                            labels.append(label_map[sentiment])
        else:
            # Fallback to CSV if available
            csv_path = data_path / "all-data.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                texts = df['text'].tolist()
                labels = df['label'].tolist()
            else:
                raise FileNotFoundError(f"No data found in {data_path}")
    else:
        # Generic news data loading
        raise NotImplementedError("Generic news data loading not implemented yet")
    
    # Split data
    from sklearn.model_selection import train_test_split
    
    if args.debug:
        # Use smaller dataset for debugging
        texts = texts[:1000]
        labels = labels[:1000]
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")
    print(f"Label distribution: {dict(zip(*np.unique(train_labels, return_counts=True)))}")
    
    # Train model
    print("Starting training...")
    history = trainer.train(train_texts, train_labels, val_texts, val_labels, tokenizer)
    
    # Evaluate model
    print("Evaluating model...")
    evaluator = NewsEvaluator(model, tokenizer)
    results = evaluator.evaluate_dataset(val_texts, val_labels)
    print("Evaluation results:", results)
    
    print("Training completed successfully!")
    
    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
