"""
Model evaluation script for news sentiment analysis.

Usage:
    python scripts/evaluate.py --model-path models/best_model.pt --config configs/bert_news.yaml
    python scripts/evaluate.py --model-path models/best_model.pt --config configs/lstm_news.yaml

Supports both transformer-based and LSTM-based models.
"""

import argparse
import torch
import yaml
import numpy as np
from pathlib import Path
import sys
import logging

sys.path.append(str(Path(__file__).parent.parent / "src"))

from models import create_model, LSTMSentimentModel
from preprocessing import NewsPreprocessor, TweetPreprocessor, VocabBuilder
from evaluation import NewsAnalyzer


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    return logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Evaluate news sentiment model')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model (.pt)')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data', type=str, default=None, help='Test data CSV (overrides config)')
    parser.add_argument('--output', type=str, default='./evaluations', help='Output directory for results')
    args = parser.parse_args()

    logger = setup_logging()
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)
    model_type = config['model'].get('type', 'standard')
    model_name = config['model']['name']
    max_length = config['model'].get('max_length', 512)

    # Load test data
    import pandas as pd
    if args.data:
        test_path = Path(args.data)
    else:
        test_path = Path(config['paths']['data_dir']) / config['paths']['dataset_name'] / 'splits/data_test.csv'
    logger.info(f"Loading test data from {test_path}")
    df = pd.read_csv(test_path)
    text_column = config['data'].get('text_column', 'text')
    label_column = config['data'].get('label_column', 'sentiment')
    texts = df[text_column].dropna().tolist()
    labels_raw = df[label_column].dropna().tolist()
    if isinstance(labels_raw[0], str):
        unique_labels = sorted(list(set(labels_raw)))
        label_map = {label: i for i, label in enumerate(unique_labels)}
        labels = [label_map[label] for label in labels_raw]
        label_names = unique_labels
    else:
        labels = labels_raw
        label_names = [str(i) for i in sorted(set(labels))]

    # Model and preprocessing
    if model_type == 'lstm':
        logger.info('Using LSTM model pipeline for evaluation')
        preprocessor = TweetPreprocessor(remove_stopwords=True)
        vocab_builder = VocabBuilder(max_vocab_size=config['model'].get('vocab_size', 5000))
        # For LSTM, you must have the vocab built from training
        # Here, we assume vocab is rebuilt from test set for demo; in production, load vocab from file
        tokenized = preprocessor.preprocess_tweets(texts)
        vocab_builder.build_vocab(tokenized)
        X = vocab_builder.encode(tokenized, max_len=config['model'].get('max_len', 50))
        y = torch.tensor(labels, dtype=torch.long)
        model = LSTMSentimentModel(
            vocab_size=len(vocab_builder.word2idx),
            embedding_dim=config['model'].get('embedding_dim', 32),
            hidden_dim=config['model'].get('hidden_dim', 32),
            output_dim=config['model'].get('num_labels', 3),
            max_len=config['model'].get('max_len', 50),
            dropout=config['model'].get('dropout', 0.4)
        )
        model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
        model.eval()
        # For LSTM, define a simple evaluator
        def predict_lstm(model, X):
            with torch.no_grad():
                logits = model(X)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1).cpu().numpy()
                confs = torch.max(probs, dim=1).values.cpu().numpy()
            return preds, confs
        preds, confs = predict_lstm(model, X)
        analyzer = NewsAnalyzer(model, vocab_builder, label_names=label_names)
        # Use analyzer for reporting
        analyzer.comprehensive_evaluation(texts, labels, output_dir=args.output)
    else:
        logger.info('Using transformer-based model pipeline for evaluation')
        preprocessor = NewsPreprocessor(tokenizer_name=model_name, max_length=max_length)
        model = create_model(config['model'])
        checkpoint = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        analyzer = NewsAnalyzer(model, tokenizer, label_names=label_names)
        analyzer.comprehensive_evaluation(texts, labels, output_dir=args.output)

    logger.info('Evaluation complete.')

if __name__ == "__main__":
    main()
