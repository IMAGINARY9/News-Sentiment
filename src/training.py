"""
Training utilities for news sentiment analysis.

Handles model training, evaluation, and optimization with support for
different training strategies and domain adaptation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from typing import Dict, List, Optional, Tuple
import wandb
from tqdm import tqdm
import logging
from pathlib import Path
import json

from models import NewsTransformerModel, NewsDataset


class NewsTrainer:
    """
    Trainer class for news sentiment analysis models.
    """
    
    def __init__(self, model: NewsTransformerModel, config: Dict, device: str = None):
        """
        Initialize trainer.
        
        Args:
            model: The model to train
            config: Training configuration
            device: Device to use (cuda/cpu)
        """
        self.model = model
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Training parameters
        self.epochs = config.get('epochs', 3)
        self.batch_size = config.get('batch_size', 16)
        self.learning_rate = config.get('learning_rate', 2e-5)
        self.weight_decay = config.get('weight_decay', 0.01)
        self.warmup_steps = config.get('warmup_steps', 0)
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        
        # Logging
        self.use_wandb = config.get('use_wandb', False)
        self.save_dir = Path(config.get('save_dir', './models'))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize optimizer and scheduler (will be set during training)
        self.optimizer = None
        self.scheduler = None
        
        self.logger = logging.getLogger(__name__)
    
    def create_data_loaders(self, train_texts: List[str], train_labels: List[int],
                          val_texts: List[str], val_labels: List[int],
                          tokenizer) -> Tuple[DataLoader, DataLoader]:
        """
        Create data loaders for training and validation.
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts
            val_labels: Validation labels
            tokenizer: Tokenizer to use
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        max_length = self.config.get('max_length', 512)
        
        train_dataset = NewsDataset(train_texts, train_labels, tokenizer, max_length)
        val_dataset = NewsDataset(val_texts, val_labels, tokenizer, max_length)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.config.get('num_workers', 4)
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.config.get('num_workers', 4)
        )
        
        return train_loader, val_loader
    
    def setup_optimizer_and_scheduler(self, train_loader: DataLoader):
        """Setup optimizer and learning rate scheduler."""
        # Calculate total training steps
        total_steps = len(train_loader) * self.epochs
        
        # Create optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Create scheduler
        if self.warmup_steps > 0:
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=total_steps
            )
        else:
            # Use cosine annealing if no warmup
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps
            )
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs['loss']
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # Update weights
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Log to wandb if enabled
            if self.use_wandb:
                wandb.log({
                    'train_loss_step': loss.item(),
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
        
        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}
    
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs['loss']
                logits = outputs['logits']
                
                total_loss += loss.item()
                
                # Get predictions
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        return {
            'val_loss': avg_loss,
            'val_accuracy': accuracy,
            'val_precision': precision,
            'val_recall': recall,
            'val_f1': f1
        }
    
    def train(self, train_texts: List[str], train_labels: List[int],
              val_texts: List[str], val_labels: List[int],
              tokenizer) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts
            val_labels: Validation labels
            tokenizer: Tokenizer to use
            
        Returns:
            Training history
        """
        self.logger.info(f"Starting training on {self.device}")
        self.logger.info(f"Training samples: {len(train_texts)}")
        self.logger.info(f"Validation samples: {len(val_texts)}")
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(
            train_texts, train_labels, val_texts, val_labels, tokenizer
        )
        
        # Setup optimizer and scheduler
        self.setup_optimizer_and_scheduler(train_loader)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': []
        }
        
        best_f1 = 0.0
        
        for epoch in range(self.epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Evaluate
            val_metrics = self.evaluate(val_loader)
            
            # Update history
            history['train_loss'].append(train_metrics['train_loss'])
            history['val_loss'].append(val_metrics['val_loss'])
            history['val_accuracy'].append(val_metrics['val_accuracy'])
            history['val_f1'].append(val_metrics['val_f1'])
            
            # Log metrics
            self.logger.info(f"Train Loss: {train_metrics['train_loss']:.4f}")
            self.logger.info(f"Val Loss: {val_metrics['val_loss']:.4f}")
            self.logger.info(f"Val Accuracy: {val_metrics['val_accuracy']:.4f}")
            self.logger.info(f"Val F1: {val_metrics['val_f1']:.4f}")
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    **train_metrics,
                    **val_metrics
                })
            
            # Save best model
            if val_metrics['val_f1'] > best_f1:
                best_f1 = val_metrics['val_f1']
                self.save_model(f'best_model_epoch_{epoch + 1}')
                self.logger.info(f"New best model saved with F1: {best_f1:.4f}")
        
        # Save final model
        self.save_model('final_model')
        
        # Save training history
        with open(self.save_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        return history
    
    def save_model(self, name: str):
        """Save model and tokenizer."""
        model_path = self.save_dir / name
        model_path.mkdir(exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'model_name': self.model.model_name,
                'num_labels': self.model.num_labels,
                'use_domain_adaptation': self.model.use_domain_adaptation
            }
        }, model_path / 'model.pt')
        
        self.logger.info(f"Model saved to {model_path}")
    
    def load_model(self, path: str):
        """Load model from path."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info(f"Model loaded from {path}")


class NewsEvaluator:
    """
    Evaluator for news sentiment analysis models.
    """
    
    def __init__(self, model: NewsTransformerModel, tokenizer, device: str = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, texts: List[str], batch_size: int = 32) -> Tuple[List[int], List[float]]:
        """
        Make predictions on texts.
        
        Args:
            texts: List of texts to predict
            batch_size: Batch size for prediction
            
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        dataset = NewsDataset(texts, [0] * len(texts), self.tokenizer)  # Dummy labels
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_predictions = []
        all_confidences = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs['logits']
                
                # Get predictions and confidence scores
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                confidences = torch.max(probabilities, dim=-1)[0]
                
                all_predictions.extend(predictions.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
        
        return all_predictions, all_confidences
    
    def evaluate_dataset(self, texts: List[str], labels: List[int]) -> Dict[str, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            texts: List of texts
            labels: True labels
            
        Returns:
            Evaluation metrics
        """
        predictions, confidences = self.predict(texts)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        
        # Generate classification report
        report = classification_report(labels, predictions, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'classification_report': report,
            'avg_confidence': np.mean(confidences)
        }
