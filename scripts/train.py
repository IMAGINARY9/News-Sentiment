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
import time
import datetime
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from preprocessing import NewsPreprocessor, TweetPreprocessor, VocabBuilder
from models import create_model, NewsDataset, LSTMSentimentModel
from training import NewsTrainer
from evaluation import NewsEvaluator
from transformers import AutoTokenizer

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging(verbose=True):
    """Setup detailed logging."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Set up console logging
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log', mode='a')
        ]
    )
    
    # Set specific loggers
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

def print_system_info():
    """Print system information for debugging."""
    print("="*80)
    print("SYSTEM INFORMATION")
    print("="*80)
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Current working directory: {Path.cwd()}")
    print("="*80)

def main():
    script_start_time = time.time()
    
    parser = argparse.ArgumentParser(description='Train news sentiment analysis model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--model', type=str, default=None, help='Model name override')
    parser.add_argument('--data', type=str, default=None, help='Data directory override')
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--debug', action='store_true', help='Debug mode with smaller dataset')
    parser.add_argument('--verbose', action='store_true', default=True, help='Verbose logging')
    parser.add_argument('--quiet', action='store_true', help='Quiet mode (minimal logging)')
    
    args = parser.parse_args()
    
    # Setup logging
    verbose = args.verbose and not args.quiet
    logger = setup_logging(verbose)
    
    # Print system info if verbose
    if verbose:
        print_system_info()
    
    # Load configuration
    config_start = time.time()
    logger.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    config_time = time.time() - config_start
    logger.info(f"Configuration loaded in {config_time:.2f}s")
    
    # Override config with CLI args
    if args.model:
        logger.info(f"Overriding model: {config['model']['name']} -> {args.model}")
        config['model']['name'] = args.model
    if args.data:
        logger.info(f"Overriding data directory: {Path(config['paths']['data_dir']) / config['paths']['dataset_name']} -> {args.data}")
        config['paths']['dataset_name'] = args.data      
    # Initialize wandb if requested
    if args.wandb:
        logger.info("Initializing Weights & Biases logging...")
        wandb.init(
            project="news-sentiment",
            config=config,
            name=f"news_sentiment_{config['model']['name'].replace('/', '_')}"
        )
        logger.info("W&B initialized successfully")
    
    logger.info("="*80)
    logger.info("TRAINING CONFIGURATION")
    logger.info("="*80)
    logger.info(f"Config file: {args.config}")
    logger.info(f"Model: {config['model']['name']}")
    logger.info(f"Data directory: {Path(config['paths']['data_dir']) / config['paths']['dataset_name']}")
    logger.info(f"Debug mode: {args.debug}")
    logger.info(f"Use W&B: {args.wandb}")
    logger.info("="*80)
    
    # Initialize tokenizer and preprocessing based on model type
    model_type = config['model'].get('type', 'standard')
    model_name = config['model']['name']
    max_length = config['model'].get('max_length', 512)

    if model_type == 'lstm':
        logger.info('Using LSTM model pipeline (tweet-style preprocessing)')
        preprocessor = TweetPreprocessor(remove_stopwords=True)
        vocab_builder = VocabBuilder(max_vocab_size=config['model'].get('vocab_size', 5000))
        # Tokenize and build vocab
        tokenized_train = preprocessor.preprocess_tweets(train_texts)
        tokenized_val = preprocessor.preprocess_tweets(val_texts)
        vocab_builder.build_vocab(tokenized_train)
        X_train = vocab_builder.encode(tokenized_train, max_len=config['model'].get('max_len', 50))
        X_val = vocab_builder.encode(tokenized_val, max_len=config['model'].get('max_len', 50))
        y_train = torch.tensor(train_labels, dtype=torch.long)
        y_val = torch.tensor(val_labels, dtype=torch.long)
        # Prepare dataset
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        # Model
        model = LSTMSentimentModel(
            vocab_size=len(vocab_builder.word2idx),
            embedding_dim=config['model'].get('embedding_dim', 32),
            hidden_dim=config['model'].get('hidden_dim', 32),
            output_dim=config['model'].get('num_labels', 3),
            max_len=config['model'].get('max_len', 50),
            dropout=config['model'].get('dropout', 0.4)
        )
        # Trainer: use a simple PyTorch training loop or a custom trainer
        trainer = NewsTrainer(
            model=model,
            config=config['training'],
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            model_type='lstm',
        )
        # Tokenizer is not used for LSTM, but pass vocab_builder if needed
        tokenizer = vocab_builder
    else:
        # Transformer-based pipeline (default)
        logger.info('Using transformer-based model pipeline')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        preprocessor = NewsPreprocessor(
            tokenizer_name=model_name,
            max_length=max_length
        )
        model = create_model(config['model'])
        trainer = NewsTrainer(
            model=model,
            config=config['training']
        )
    
    # Load and preprocess data
    data_start = time.time()
    logger.info("="*60)
    logger.info("DATA LOADING (prepared splits)")
    logger.info("="*60)
    logger.info("Loading prepared train/val splits...")
    data_path = Path(config['paths']['data_dir']) / config['paths']['dataset_name']
    train_file = data_path / "splits/data_train.csv"
    val_file = data_path / "splits/data_val.csv"
    if not train_file.exists() or not val_file.exists():
        raise FileNotFoundError(f"Prepared split files not found: {train_file} or {val_file}. Please run the data preparation notebook to generate these files.")
    import pandas as pd
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    text_column = config['data'].get('text_column', 'text')
    label_column = config['data'].get('label_column', 'sentiment')
    train_texts = train_df[text_column].dropna().tolist()
    train_labels_raw = train_df[label_column].dropna().tolist()
    val_texts = val_df[text_column].dropna().tolist()
    val_labels_raw = val_df[label_column].dropna().tolist()
    # Map labels to integers if needed
    if isinstance(train_labels_raw[0], str):
        unique_labels = sorted(list(set(train_labels_raw + val_labels_raw)))
        label_map = {label: i for i, label in enumerate(unique_labels)}
        train_labels = [label_map[label] for label in train_labels_raw]
        val_labels = [label_map[label] for label in val_labels_raw]
        logger.info(f"Label mapping: {label_map}")
    else:
        train_labels = train_labels_raw
        val_labels = val_labels_raw
    logger.info(f"Loaded {len(train_texts):,} training samples, {len(val_texts):,} validation samples")
    data_time = time.time() - data_start
    logger.info(f"Data loading completed in {data_time:.2f}s")
    
    # Data is already split and loaded above, so no need to split again
    # Remove redundant split logic
    # Remove: from sklearn.model_selection import train_test_split
    # Remove: if args.debug: ...
    # Remove: split_start = time.time() ... train_test_split ... split_time = ...
    # Instead, just log the label distribution and average text length
    label_dist = dict(zip(*np.unique(train_labels, return_counts=True)))
    logger.info(f"Label distribution: {label_dist}")
    avg_text_length = np.mean([len(text.split()) for text in train_texts])
    logger.info(f"Average text length: {avg_text_length:.1f} words")
    
    # Train model
    training_start = time.time()
    logger.info("="*80)
    logger.info("STARTING MODEL TRAINING")
    logger.info("="*80)
    
    # For LSTM, pass datasets directly; for transformers, use NewsDataset
    if model_type == 'lstm':
        history = trainer.train(None, None, None, None, tokenizer)
    else:
        history = trainer.train(train_texts, train_labels, val_texts, val_labels, tokenizer)
    
    training_time = time.time() - training_start
    
    # Evaluate model
    eval_start = time.time()
    logger.info("="*60)
    logger.info("FINAL MODEL EVALUATION")
    logger.info("="*60)
    logger.info("Evaluating model on validation set...")
    
    if model_type == 'lstm':
        evaluator = NewsEvaluator(model, tokenizer, model_type='lstm')
        results = evaluator.evaluate_dataset(val_dataset)
    else:
        evaluator = NewsEvaluator(model, tokenizer)
        results = evaluator.evaluate_dataset(val_texts, val_labels)
    
    eval_time = time.time() - eval_start
    logger.info(f"Final evaluation completed in {eval_time:.2f}s")
    
    # Print comprehensive results
    logger.info("="*80)
    logger.info("FINAL RESULTS")
    logger.info("="*80)
    logger.info(f"Accuracy: {results['accuracy']:.4f}")
    logger.info(f"Precision: {results['precision']:.4f}")
    logger.info(f"Recall: {results['recall']:.4f}")
    logger.info(f"F1 Score: {results['f1']:.4f}")
    logger.info(f"Average Confidence: {results['avg_confidence']:.4f}")
    
    # Total script time
    total_time = time.time() - script_start_time
    logger.info("="*80)
    logger.info("TIMING SUMMARY")
    logger.info("="*80)
    logger.info(f"Total script time: {str(datetime.timedelta(seconds=int(total_time)))}")
    logger.info(f"Data loading: {data_time:.2f}s")
    logger.info(f"Model training: {training_time:.2f}s")
    logger.info(f"Final evaluation: {eval_time:.2f}s")
    logger.info("="*80)
    
    logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY! 🎉")
    
    if args.wandb:
        logger.info("Finishing W&B session...")
        wandb.finish()

if __name__ == "__main__":
    main()