"""
Preprocessing utilities for news sentiment analysis.

Handles text cleaning, tokenization, and preparation for long documents.
"""

import re
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from transformers import AutoTokenizer
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
import demoji
import spacy

class NewsPreprocessor:
    """Preprocessor for news articles with sentiment analysis focus."""
    
    def __init__(self, tokenizer_name: str = "bert-base-uncased", max_length: int = 512):
        """
        Initialize the preprocessor.
        
        Args:
            tokenizer_name: Name of the tokenizer to use
            max_length: Maximum sequence length for tokenization
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        self.stop_words = set(stopwords.words('english'))
        
        # Load spaCy model for advanced preprocessing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy English model not found. Some features may be limited.")
            self.nlp = None
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove emojis (convert to text first if needed)
        text = demoji.replace(text, '')
        
        return text.strip()
    
    def chunk_long_text(self, text: str, max_chunk_length: int = 400, overlap: int = 50) -> List[str]:
        """
        Split long text into overlapping chunks for processing.
        
        Args:
            text: Text to chunk
            max_chunk_length: Maximum tokens per chunk
            overlap: Number of overlapping tokens between chunks
            
        Returns:
            List of text chunks
        """
        # Tokenize the full text
        tokens = self.tokenizer.tokenize(text)
        
        if len(tokens) <= max_chunk_length:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(tokens):
            end = min(start + max_chunk_length, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.convert_tokens_to_string(chunk_tokens)
            chunks.append(chunk_text)
            
            if end >= len(tokens):
                break
                
            start = end - overlap
        
        return chunks
    
    def preprocess_article(self, article: str, chunk_long_articles: bool = True) -> Dict:
        """
        Preprocess a single news article.
        
        Args:
            article: Raw article text
            chunk_long_articles: Whether to chunk long articles
            
        Returns:
            Dictionary with processed text and metadata
        """
        # Clean the text
        cleaned_text = self.clean_text(article)
        
        if not cleaned_text:
            return {
                'text': '',
                'chunks': [],
                'is_long': False,
                'num_tokens': 0
            }
        
        # Get token count
        tokens = self.tokenizer.tokenize(cleaned_text)
        num_tokens = len(tokens)
        is_long = num_tokens > self.max_length
        
        result = {
            'text': cleaned_text,
            'is_long': is_long,
            'num_tokens': num_tokens
        }
        
        # Handle long articles
        if is_long and chunk_long_articles:
            chunks = self.chunk_long_text(cleaned_text)
            result['chunks'] = chunks
        else:
            result['chunks'] = [cleaned_text]
        
        return result
    
    def preprocess_dataset(self, df: pd.DataFrame, text_column: str = 'text', 
                          sentiment_column: Optional[str] = None) -> pd.DataFrame:
        """
        Preprocess an entire dataset.
        
        Args:
            df: DataFrame with news articles
            text_column: Name of the text column
            sentiment_column: Name of the sentiment column (if exists)
            
        Returns:
            Preprocessed DataFrame
        """
        processed_data = []
        
        for idx, row in df.iterrows():
            article_text = row[text_column]
            processed = self.preprocess_article(article_text)
            
            # Create entries for each chunk
            for chunk_idx, chunk in enumerate(processed['chunks']):
                entry = {
                    'original_id': idx,
                    'chunk_id': chunk_idx,
                    'text': chunk,
                    'is_long': processed['is_long'],
                    'total_chunks': len(processed['chunks']),
                    'num_tokens': processed['num_tokens']
                }
                
                # Add sentiment if available
                if sentiment_column and sentiment_column in row:
                    entry['sentiment'] = row[sentiment_column]
                
                # Add other columns
                for col in df.columns:
                    if col not in [text_column, sentiment_column]:
                        entry[col] = row[col]
                
                processed_data.append(entry)
        
        return pd.DataFrame(processed_data)
    
    def tokenize_for_model(self, texts: List[str]) -> Dict:
        """
        Tokenize texts for model input.
        
        Args:
            texts: List of texts to tokenize
            
        Returns:
            Tokenized inputs ready for model
        """
        return self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
