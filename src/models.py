"""
Model definitions for news sentiment analysis.

Implements transformer-based models for news sentiment classification
with support for different pre-trained models and domain-specific fine-tuning.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    TrainingArguments, Trainer,
    get_linear_schedule_with_warmup
)
from typing import Dict, List, Optional, Tuple
import numpy as np
from torch.utils.data import Dataset


class NewsDataset(Dataset):
    """Dataset class for news sentiment data."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class NewsTransformerModel(nn.Module):
    """
    Transformer-based model for news sentiment analysis.
    
    Supports various pre-trained models including FinBERT for financial news.
    """
    
    def __init__(self, model_name: str = "bert-base-uncased", num_labels: int = 3, 
                 dropout_rate: float = 0.1, use_domain_adaptation: bool = False):
        """
        Initialize the model.
        
        Args:
            model_name: Pre-trained model name (bert-base-uncased, distilbert, finbert, etc.)
            num_labels: Number of sentiment classes (3 for negative/neutral/positive)
            dropout_rate: Dropout rate for regularization
            use_domain_adaptation: Whether to use domain adaptation layers
        """
        super(NewsTransformerModel, self).__init__()
        
        self.model_name = model_name
        self.num_labels = num_labels
        self.use_domain_adaptation = use_domain_adaptation
        
        # Load pre-trained model and config
        self.config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        self.bert = AutoModel.from_pretrained(model_name, config=self.config)
        
        # Get hidden size from model config
        self.hidden_size = self.bert.config.hidden_size
        
        # Domain adaptation layers (if enabled)
        if use_domain_adaptation:
            self.domain_adapter = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.hidden_size // 2, self.hidden_size)
            )
        
        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size // 2, num_labels)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier weights."""
        for module in [self.classifier]:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        torch.nn.init.xavier_uniform_(layer.weight)
                        layer.bias.data.fill_(0.01)
    
    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            labels: Ground truth labels (optional)
            
        Returns:
            Dictionary with loss (if labels provided) and logits
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        
        # Apply domain adaptation if enabled
        if self.use_domain_adaptation:
            pooled_output = self.domain_adapter(pooled_output) + pooled_output  # Residual connection
        
        # Apply dropout and classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        result = {"logits": logits}
        
        # Calculate loss if labels are provided
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            result["loss"] = loss
        
        return result
    
    def get_embeddings(self, input_ids, attention_mask):
        """Get embeddings for analysis."""
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.last_hidden_state, outputs.pooler_output


class FinancialNewsModel(NewsTransformerModel):
    """
    Specialized model for financial news sentiment analysis.
    
    Uses FinBERT as the base model and includes financial domain-specific features.
    """
    
    def __init__(self, num_labels: int = 3, dropout_rate: float = 0.1):
        # Use FinBERT for financial domain
        super().__init__(
            model_name="ProsusAI/finbert",
            num_labels=num_labels,
            dropout_rate=dropout_rate,
            use_domain_adaptation=True
        )
        
        # Financial-specific features
        self.financial_keywords = [
            'revenue', 'profit', 'earnings', 'sales', 'growth', 'decline',
            'increase', 'decrease', 'bullish', 'bearish', 'volatility',
            'market', 'stock', 'share', 'dividend', 'invest'
        ]
        
    def extract_financial_features(self, texts: List[str]) -> torch.Tensor:
        """Extract financial keyword features."""
        features = []
        for text in texts:
            text_lower = text.lower()
            keyword_count = sum(1 for keyword in self.financial_keywords if keyword in text_lower)
            features.append(keyword_count / len(self.financial_keywords))  # Normalize
        return torch.tensor(features, dtype=torch.float32)


class LongDocumentModel(NewsTransformerModel):
    """
    Model for handling long news documents using hierarchical attention.
    """
    
    def __init__(self, model_name: str = "bert-base-uncased", num_labels: int = 3,
                 max_chunks: int = 10, chunk_size: int = 512):
        super().__init__(model_name=model_name, num_labels=num_labels)
        
        self.max_chunks = max_chunks
        self.chunk_size = chunk_size
        
        # Chunk-level attention
        self.chunk_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            dropout=0.1
        )
        
        # Document-level aggregation
        self.document_aggregator = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, chunk_input_ids, chunk_attention_masks, labels=None):
        """
        Forward pass for long documents.
        
        Args:
            chunk_input_ids: List of input_ids for each chunk [batch_size, num_chunks, seq_len]
            chunk_attention_masks: List of attention masks for each chunk
            labels: Ground truth labels
            
        Returns:
            Dictionary with loss and logits
        """
        batch_size, num_chunks, seq_len = chunk_input_ids.shape
        
        # Process each chunk
        chunk_representations = []
        for i in range(num_chunks):
            chunk_outputs = self.bert(
                input_ids=chunk_input_ids[:, i, :],
                attention_mask=chunk_attention_masks[:, i, :]
            )
            chunk_representations.append(chunk_outputs.pooler_output)
        
        # Stack chunk representations [batch_size, num_chunks, hidden_size]
        chunk_repr = torch.stack(chunk_representations, dim=1)
        
        # Apply attention across chunks
        attended_chunks, _ = self.chunk_attention(
            chunk_repr.transpose(0, 1),  # [num_chunks, batch_size, hidden_size]
            chunk_repr.transpose(0, 1),
            chunk_repr.transpose(0, 1)
        )
        
        # Aggregate to document level
        document_repr = torch.mean(attended_chunks.transpose(0, 1), dim=1)
        document_repr = self.document_aggregator(document_repr)
        
        # Final classification
        logits = self.classifier(self.dropout(document_repr))
        
        result = {"logits": logits}
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            result["loss"] = loss
        
        return result


class LSTMSentimentModel(nn.Module):
    """
    Bidirectional LSTM model for tweet sentiment classification (PyTorch version).
    """
    def __init__(self, vocab_size, embedding_dim=32, hidden_dim=32, output_dim=3, max_len=50, dropout=0.4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv1d = nn.Conv1d(embedding_dim, 32, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(32, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.max_len = max_len

    def forward(self, x):
        # x: [batch_size, seq_len]
        x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        x = x.transpose(1, 2)  # [batch_size, embedding_dim, seq_len]
        x = self.conv1d(x)     # [batch_size, 32, seq_len]
        x = self.maxpool(x)    # [batch_size, 32, seq_len//2]
        x = x.transpose(1, 2)  # [batch_size, seq_len//2, 32]
        output, (h_n, c_n) = self.lstm(x)  # output: [batch_size, seq_len//2, hidden_dim*2]
        out = output[:, -1, :]  # Take last output
        out = self.dropout(out)
        logits = self.fc(out)
        return logits


def create_model(config: Dict) -> NewsTransformerModel:
    """
    Factory function to create models based on configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Initialized model
    """
    model_type = config.get('type', 'standard')
    model_name = config.get('name', 'bert-base-uncased')
    num_labels = config.get('num_labels', 3)
    dropout_rate = config.get('dropout_rate', 0.1)
    
    if model_type == 'financial':
        return FinancialNewsModel(num_labels=num_labels, dropout_rate=dropout_rate)
    elif model_type == 'long_document':
        return LongDocumentModel(
            model_name=model_name,
            num_labels=num_labels,
            max_chunks=config.get('max_chunks', 10),
            chunk_size=config.get('chunk_size', 512)
        )
    else:
        return NewsTransformerModel(
            model_name=model_name,
            num_labels=num_labels,
            dropout_rate=dropout_rate,
            use_domain_adaptation=config.get('use_domain_adaptation', False)
        )
