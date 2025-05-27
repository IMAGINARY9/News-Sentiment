"""
Evaluation utilities for news sentiment analysis.

Provides comprehensive evaluation metrics, visualizations, and analysis tools
for news sentiment classification models.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path
import logging
from collections import Counter

from models import NewsTransformerModel
from training import NewsEvaluator


class NewsAnalyzer:
    """
    Advanced analyzer for news sentiment models.
    """
    
    def __init__(self, model: NewsTransformerModel, tokenizer, label_names: List[str] = None):
        """
        Initialize analyzer.
        
        Args:
            model: Trained model
            tokenizer: Tokenizer used for the model
            label_names: Names for sentiment labels (e.g., ['negative', 'neutral', 'positive'])
        """
        self.model = model
        self.tokenizer = tokenizer
        self.evaluator = NewsEvaluator(model, tokenizer)
        self.label_names = label_names or ['negative', 'neutral', 'positive']
        self.logger = logging.getLogger(__name__)
    
    def comprehensive_evaluation(self, texts: List[str], labels: List[int],
                                output_dir: str = './evaluation_results') -> Dict:
        """
        Perform comprehensive evaluation with visualizations.
        
        Args:
            texts: Test texts
            labels: True labels
            output_dir: Directory to save results
            
        Returns:
            Comprehensive evaluation results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Starting comprehensive evaluation...")
        
        # Get predictions
        predictions, confidences = self.evaluator.predict(texts)
        
        # Basic metrics
        metrics = self._calculate_metrics(labels, predictions, confidences)
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        self._plot_confusion_matrix(cm, output_path / 'confusion_matrix.png')
        
        # Classification report
        report = classification_report(labels, predictions, 
                                     target_names=self.label_names, 
                                     output_dict=True)
        
        # ROC curves (for multi-class)
        if len(set(labels)) > 2:
            self._plot_roc_curves(labels, predictions, output_path / 'roc_curves.png')
        
        # Confidence distribution
        self._plot_confidence_distribution(confidences, predictions, labels,
                                         output_path / 'confidence_distribution.png')
        
        # Error analysis
        error_analysis = self._analyze_errors(texts, labels, predictions, confidences)
        
        # Sentiment distribution
        self._plot_sentiment_distribution(labels, predictions, 
                                        output_path / 'sentiment_distribution.png')
        
        # Length analysis
        length_analysis = self._analyze_by_length(texts, labels, predictions)
        
        # Compile results
        results = {
            'metrics': metrics,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'error_analysis': error_analysis,
            'length_analysis': length_analysis,
            'label_names': self.label_names
        }
        
        # Save results
        with open(output_path / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Evaluation complete. Results saved to {output_path}")
        return results
    
    def _calculate_metrics(self, labels: List[int], predictions: List[int], 
                          confidences: List[float]) -> Dict:
        """Calculate comprehensive metrics."""
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support = \
            precision_recall_fscore_support(labels, predictions, average=None)
        
        return {
            'accuracy': accuracy,
            'weighted_precision': precision,
            'weighted_recall': recall,
            'weighted_f1': f1,
            'per_class_precision': precision_per_class.tolist(),
            'per_class_recall': recall_per_class.tolist(),
            'per_class_f1': f1_per_class.tolist(),
            'support': support.tolist(),
            'avg_confidence': np.mean(confidences),
            'confidence_std': np.std(confidences)
        }
    
    def _plot_confusion_matrix(self, cm: np.ndarray, save_path: Path):
        """Plot and save confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_names,
                   yticklabels=self.label_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curves(self, labels: List[int], predictions: List[int], save_path: Path):
        """Plot ROC curves for multi-class classification."""
        # Binarize labels for multi-class ROC
        y_true_bin = label_binarize(labels, classes=list(range(len(self.label_names))))
        y_pred_bin = label_binarize(predictions, classes=list(range(len(self.label_names))))
        
        plt.figure(figsize=(10, 8))
        
        for i, label_name in enumerate(self.label_names):
            if y_true_bin.shape[1] > i:  # Check if class exists
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
                auc = roc_auc_score(y_true_bin[:, i], y_pred_bin[:, i])
                plt.plot(fpr, tpr, label=f'{label_name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confidence_distribution(self, confidences: List[float], 
                                    predictions: List[int], labels: List[int],
                                    save_path: Path):
        """Plot confidence score distributions."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Overall confidence distribution
        axes[0, 0].hist(confidences, bins=50, alpha=0.7, color='blue')
        axes[0, 0].set_title('Overall Confidence Distribution')
        axes[0, 0].set_xlabel('Confidence Score')
        axes[0, 0].set_ylabel('Frequency')
        
        # Confidence by correctness
        correct_mask = np.array(predictions) == np.array(labels)
        correct_conf = np.array(confidences)[correct_mask]
        incorrect_conf = np.array(confidences)[~correct_mask]
        
        axes[0, 1].hist(correct_conf, bins=30, alpha=0.7, label='Correct', color='green')
        axes[0, 1].hist(incorrect_conf, bins=30, alpha=0.7, label='Incorrect', color='red')
        axes[0, 1].set_title('Confidence by Correctness')
        axes[0, 1].set_xlabel('Confidence Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        
        # Confidence by predicted class
        for i, label_name in enumerate(self.label_names):
            class_mask = np.array(predictions) == i
            if np.any(class_mask):
                class_conf = np.array(confidences)[class_mask]
                axes[1, 0].hist(class_conf, bins=20, alpha=0.7, label=label_name)
        axes[1, 0].set_title('Confidence by Predicted Class')
        axes[1, 0].set_xlabel('Confidence Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # Confidence vs Accuracy scatter
        unique_conf = np.linspace(0, 1, 20)
        accuracies = []
        for conf_threshold in unique_conf:
            mask = np.array(confidences) >= conf_threshold
            if np.any(mask):
                acc = accuracy_score(np.array(labels)[mask], np.array(predictions)[mask])
                accuracies.append(acc)
            else:
                accuracies.append(0)
        
        axes[1, 1].plot(unique_conf, accuracies, 'o-')
        axes[1, 1].set_title('Accuracy vs Confidence Threshold')
        axes[1, 1].set_xlabel('Confidence Threshold')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _analyze_errors(self, texts: List[str], labels: List[int], 
                       predictions: List[int], confidences: List[float]) -> Dict:
        """Analyze prediction errors."""
        # Find misclassified samples
        error_mask = np.array(predictions) != np.array(labels)
        error_indices = np.where(error_mask)[0]
        
        # Error analysis by class
        error_by_class = {}
        for i, label_name in enumerate(self.label_names):
            true_class_mask = np.array(labels) == i
            class_errors = error_mask & true_class_mask
            error_count = np.sum(class_errors)
            total_count = np.sum(true_class_mask)
            
            error_by_class[label_name] = {
                'error_count': int(error_count),
                'total_count': int(total_count),
                'error_rate': float(error_count / total_count) if total_count > 0 else 0.0
            }
        
        # Most confident errors
        error_confidences = np.array(confidences)[error_mask]
        if len(error_confidences) > 0:
            top_confident_errors_idx = error_indices[np.argsort(error_confidences)[-10:]]
            most_confident_errors = [
                {
                    'text': texts[idx][:200] + "..." if len(texts[idx]) > 200 else texts[idx],
                    'true_label': self.label_names[labels[idx]],
                    'predicted_label': self.label_names[predictions[idx]],
                    'confidence': float(confidences[idx])
                }
                for idx in top_confident_errors_idx
            ]
        else:
            most_confident_errors = []
        
        return {
            'total_errors': int(np.sum(error_mask)),
            'error_rate': float(np.mean(error_mask)),
            'error_by_class': error_by_class,
            'most_confident_errors': most_confident_errors
        }
    
    def _plot_sentiment_distribution(self, labels: List[int], predictions: List[int], 
                                   save_path: Path):
        """Plot sentiment distribution comparison."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # True distribution
        true_counts = Counter(labels)
        true_labels = [self.label_names[i] for i in sorted(true_counts.keys())]
        true_values = [true_counts[i] for i in sorted(true_counts.keys())]
        
        ax1.bar(true_labels, true_values, color='lightblue', alpha=0.7)
        ax1.set_title('True Sentiment Distribution')
        ax1.set_ylabel('Count')
        
        # Predicted distribution
        pred_counts = Counter(predictions)
        pred_labels = [self.label_names[i] for i in sorted(pred_counts.keys())]
        pred_values = [pred_counts[i] for i in sorted(pred_counts.keys())]
        
        ax2.bar(pred_labels, pred_values, color='lightcoral', alpha=0.7)
        ax2.set_title('Predicted Sentiment Distribution')
        ax2.set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _analyze_by_length(self, texts: List[str], labels: List[int], 
                          predictions: List[int]) -> Dict:
        """Analyze performance by text length."""
        lengths = [len(text.split()) for text in texts]
        
        # Define length bins
        length_bins = [(0, 50), (50, 150), (150, 300), (300, float('inf'))]
        bin_names = ['Short (0-50)', 'Medium (50-150)', 'Long (150-300)', 'Very Long (300+)']
        
        length_analysis = {}
        for (min_len, max_len), bin_name in zip(length_bins, bin_names):
            mask = (np.array(lengths) >= min_len) & (np.array(lengths) < max_len)
            if np.any(mask):
                bin_labels = np.array(labels)[mask]
                bin_predictions = np.array(predictions)[mask]
                accuracy = accuracy_score(bin_labels, bin_predictions)
                
                length_analysis[bin_name] = {
                    'count': int(np.sum(mask)),
                    'accuracy': float(accuracy),
                    'avg_length': float(np.mean(np.array(lengths)[mask]))
                }
            else:
                length_analysis[bin_name] = {
                    'count': 0,
                    'accuracy': 0.0,
                    'avg_length': 0.0
                }
        
        return length_analysis


def evaluate_financial_sentiment(model_path: str, test_data_path: str, 
                                output_dir: str = './financial_evaluation'):
    """
    Specialized evaluation for financial sentiment models.
    
    Args:
        model_path: Path to trained model
        test_data_path: Path to test data
        output_dir: Output directory for results
    """
    # Load model and data
    # Implementation would load the specific financial model
    # and perform domain-specific evaluation
    pass
