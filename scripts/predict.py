"""
Script for predicting sentiment of raw news text using a trained model.

Usage:
    python scripts/predict.py --model-path models/best_model_epoch_1/model.pt --config configs/base_config.yaml --text "Your news text here"
    python scripts/predict.py --model-path models/best_model_epoch_1/model.pt --config configs/base_config.yaml --file input.txt

Supports both transformer-based and LSTM-based models.
"""

import argparse
import torch
import yaml
from pathlib import Path
import sys
import logging

sys.path.append(str(Path(__file__).parent.parent / "src"))

from models import create_model, LSTMSentimentModel
from preprocessing import NewsPreprocessor, TweetPreprocessor, VocabBuilder
from transformers import AutoTokenizer


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    return logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Predict sentiment for raw news text')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model (.pt)')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--text', type=str, default=None, help='Raw text to predict')
    parser.add_argument('--file', type=str, default=None, help='Text file with one document per line')
    parser.add_argument('--visualize', action='store_true', help='If set, generate and save prediction explanation plots')
    args = parser.parse_args()

    logger = setup_logging()
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)
    model_type = config['model'].get('type', 'standard')
    model_name = config['model']['name']
    max_length = config['model'].get('max_length', 512)

    # Load input text(s)
    if args.text:
        texts = [args.text]
    elif args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        logger.error('Please provide --text or --file')
        sys.exit(1)

    # Label mapping (default)
    label_names = config['data'].get('label_names', ['negative', 'neutral', 'positive'])

    # Import visualization utilities if needed
    if args.visualize:
        from visualization import explain_and_plot_transformer, explain_and_plot_lstm
        import os
        vis_dir = os.path.join(os.path.dirname(__file__), '..', 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)

    if model_type == 'lstm':
        logger.info('Using LSTM model pipeline for prediction')
        preprocessor = TweetPreprocessor(remove_stopwords=True)
        vocab_builder = VocabBuilder(max_vocab_size=config['model'].get('vocab_size', 5000))
        tokenized = preprocessor.preprocess_tweets(texts)
        vocab_builder.build_vocab(tokenized)
        X = vocab_builder.encode(tokenized, max_len=config['model'].get('max_len', 50))
        model = LSTMSentimentModel(
            vocab_size=len(vocab_builder.word2idx),
            embedding_dim=config['model'].get('embedding_dim', 32),
            hidden_dim=config['model'].get('hidden_dim', 32),
            output_dim=config['model'].get('num_labels', 3),
            max_len=config['model'].get('max_len', 50),
            dropout=config['model'].get('dropout', 0.4)
        )
        checkpoint = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        with torch.no_grad():
            logits = model(X)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            confs = torch.max(probs, dim=1).values.cpu().numpy()
        for i, text in enumerate(texts):
            print(f"Text: {text}")
            print(f"Predicted label: {label_names[preds[i]] if preds[i] < len(label_names) else preds[i]} (confidence: {confs[i]:.2f})\n")
            if args.visualize:
                save_path = os.path.join(vis_dir, f"lstm_explain_{i}.png")
                try:
                    explain_and_plot_lstm(model, vocab_builder, text, label_names, save_path)
                    print(f"Visualization saved to {save_path}")
                except Exception as e:
                    print(f"Visualization failed: {e}")
    else:
        logger.info('Using transformer-based model pipeline for prediction')
        preprocessor = NewsPreprocessor(tokenizer_name=model_name, max_length=max_length)
        model = create_model(config['model'])
        checkpoint = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Preprocess and tokenize
        encodings = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        with torch.no_grad():
            outputs = model(encodings['input_ids'], encodings['attention_mask'])
            logits = outputs['logits']
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            confs = torch.max(probs, dim=1).values.cpu().numpy()
        for i, text in enumerate(texts):
            print(f"Text: {text}")
            print(f"Predicted label: {label_names[preds[i]] if preds[i] < len(label_names) else preds[i]} (confidence: {confs[i]:.2f})\n")
            if args.visualize:
                save_path = os.path.join(vis_dir, f"transformer_explain_{i}.png")
                try:
                    explain_and_plot_transformer(model, tokenizer, text, label_names, save_path)
                    print(f"Visualization saved to {save_path}")
                except Exception as e:
                    print(f"Visualization failed: {e}")

if __name__ == "__main__":
    main()
