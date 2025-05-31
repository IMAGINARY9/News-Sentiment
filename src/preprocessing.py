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
from nltk.stem.porter import PorterStemmer
from collections import Counter
import torch

def detect_text_and_label_columns(df, text_candidates=None, label_candidates=None):
    """Utility to detect text and label columns in a DataFrame."""
    if text_candidates is None:
        text_candidates = ['text', 'content', 'article', 'news', 'sentence', 'clean_text']
    if label_candidates is None:
        label_candidates = ['sentiment', 'label', 'target', 'class', 'category']
    text_col = None
    label_col = None
    for col in text_candidates:
        if col in df.columns:
            text_col = col
            break
    for col in label_candidates:
        if col in df.columns:
            label_col = col
            break
    return text_col, label_col

class NewsPreprocessor:
    """Preprocessor for news articles with sentiment analysis focus."""
    
    def __init__(self, tokenizer_name: str = "bert-base-uncased", max_length: int = 512,
                 remove_stopwords: bool = False, min_chunk_tokens: int = 5,
                 max_chunk_length: int = 400, chunk_overlap: int = 50,
                 emoji_strategy: str = "remove", clean_punctuation: bool = True):
        """
        Initialize the preprocessor.
        
        Args:
            tokenizer_name: Name of the tokenizer to use
            max_length: Maximum sequence length for tokenization
            remove_stopwords: Whether to remove stopwords during preprocessing
            min_chunk_tokens: Minimum tokens required for a chunk to be kept
            max_chunk_length: Maximum tokens per chunk when splitting long texts
            chunk_overlap: Number of overlapping tokens between chunks
            emoji_strategy: How to handle emojis ("remove", "replace", "count")
            clean_punctuation: Whether to remove punctuation during preprocessing
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.remove_stopwords = remove_stopwords
        self.min_chunk_tokens = min_chunk_tokens
        self.max_chunk_length = max_chunk_length
        self.chunk_overlap = chunk_overlap
        self.emoji_strategy = emoji_strategy
        self.clean_punctuation = clean_punctuation
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        if self.remove_stopwords:
            self.stop_words = set(stopwords.words('english'))
        else:
            self.stop_words = set()
        
        # Load spaCy model for advanced preprocessing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy English model not found. Some features may be limited.")
            self.nlp = None    
    def _remove_stopwords(self, text: str) -> str:
        """Remove stopwords from text if enabled."""
        if not self.remove_stopwords or not self.stop_words:
            return text
        
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        return ' '.join(filtered_words)
    
    def clean_text(self, text: str) -> Dict:
        """
        Clean and normalize text with detailed metrics, including punctuation cleaning if enabled.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Dictionary with cleaned text and cleaning metrics
        """
        if not isinstance(text, str):
            return {
                'cleaned_text': '',
                'metrics': {
                    'urls_removed': 0,
                    'html_tags_removed': 0,
                    'emails_removed': 0,
                    'emojis_removed': 0,
                    'punctuation_removed': 0,
                    'chars_removed': 0,
                    'stopwords_removed': 0
                }
            }
        
        original_length = len(text)
        metrics = {
            'urls_removed': 0,
            'html_tags_removed': 0,
            'emails_removed': 0,
            'emojis_removed': 0,
            'punctuation_removed': 0,
            'chars_removed': 0,
            'stopwords_removed': 0
        }
        
        # Remove HTML tags
        html_pattern = r'<.*?>'
        metrics['html_tags_removed'] = len(re.findall(html_pattern, text))
        text = re.sub(html_pattern, '', text)
        
        # Remove URLs
        url_pattern = r'https?://\S+|www\.\S+'
        metrics['urls_removed'] = len(re.findall(url_pattern, text))
        text = re.sub(url_pattern, '', text)
        
        # Remove email addresses
        email_pattern = r'\S+@\S+'
        metrics['emails_removed'] = len(re.findall(email_pattern, text))
        text = re.sub(email_pattern, '', text)
        
        # Handle emojis based on strategy
        if self.emoji_strategy == "replace":
            # Count emojis before replacement
            import emoji
            emoji_count = len([c for c in text if c in emoji.EMOJI_DATA])
            metrics['emojis_removed'] = emoji_count
            text = demoji.replace(text, ' <EMOJI> ')
        elif self.emoji_strategy == "count":
            # Count and remove emojis
            import emoji
            emoji_count = len([c for c in text if c in emoji.EMOJI_DATA])
            metrics['emojis_removed'] = emoji_count
            text = demoji.replace(text, '')
        else:  # remove
            # Just remove emojis (default behavior)
            original_text = text
            text = demoji.replace(text, '')
            # Estimate emoji count by character difference
            metrics['emojis_removed'] = max(0, len(original_text) - len(text))
        
        # Remove punctuation if enabled
        if self.clean_punctuation:
            import string
            before = len(text)
            text = text.translate(str.maketrans('', '', string.punctuation))
            after = len(text)
            metrics['punctuation_removed'] = before - after
        else:
            metrics['punctuation_removed'] = 0
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove stopwords if enabled
        if self.remove_stopwords:
            words_before = len(text.split())
            text = self._remove_stopwords(text)
            words_after = len(text.split())
            metrics['stopwords_removed'] = words_before - words_after
        
        # Calculate total character reduction
        metrics['chars_removed'] = original_length - len(text.strip())
        
        return {
            'cleaned_text': text.strip(),
            'metrics': metrics
        }
    
    def chunk_long_text(self, text: str) -> List[str]:
        """
        Split long text into overlapping chunks for processing.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks that meet minimum token requirements
        """
        # Tokenize the full text
        tokens = self.tokenizer.tokenize(text)
        
        if len(tokens) <= self.max_chunk_length:
            # Check if single text meets minimum requirements
            if len(tokens) >= self.min_chunk_tokens:
                return [text]
            else:
                return []  # Text too short
        
        chunks = []
        start = 0
        
        while start < len(tokens):
            end = min(start + self.max_chunk_length, len(tokens))
            chunk_tokens = tokens[start:end]
            
            # Only include chunk if it meets minimum token requirements
            if len(chunk_tokens) >= self.min_chunk_tokens:
                chunk_text = self.tokenizer.convert_tokens_to_string(chunk_tokens)
                chunks.append(chunk_text)
            
            if end < len(tokens):
                start = end - self.chunk_overlap
            else:
                break
        return chunks
    
    def preprocess_article(self, article: str, chunk_long_articles: bool = True) -> Dict:
        """
        Preprocess a single news article.
        
        Args:
            article: Raw article text
            chunk_long_articles: Whether to chunk long articles
            
        Returns:
            Dictionary with processed text, chunks, and cleaning metrics
        """
        # Clean the text
        cleaning_result = self.clean_text(article)
        cleaned_text = cleaning_result['cleaned_text']
        cleaning_metrics = cleaning_result['metrics']
        
        if not cleaned_text:
            return {
                'text': '',
                'chunks': [],
                'is_long': False,
                'num_tokens': 0,
                'cleaning_metrics': cleaning_metrics,
                'filtered_short_chunks': 0
            }
        
        # Get token count
        tokens = self.tokenizer.tokenize(cleaned_text)
        num_tokens = len(tokens)
        is_long = num_tokens > self.max_length
        
        result = {
            'text': cleaned_text,
            'is_long': is_long,
            'num_tokens': num_tokens,
            'cleaning_metrics': cleaning_metrics
        }
        
        # Handle long articles
        if is_long and chunk_long_articles:
            chunks = self.chunk_long_text(cleaned_text)
            result['chunks'] = chunks
            # Calculate how many chunks were filtered out
            total_potential_chunks = max(1, (num_tokens - self.chunk_overlap) // (self.max_chunk_length - self.chunk_overlap) + 1)
            result['filtered_short_chunks'] = max(0, total_potential_chunks - len(chunks))
        else:
            # For non-long articles, still check minimum token requirement
            if num_tokens >= self.min_chunk_tokens:
                result['chunks'] = [cleaned_text]
                result['filtered_short_chunks'] = 0
            else:
                result['chunks'] = []
                result['filtered_short_chunks'] = 1
        
        return result    
    def preprocess_dataset(self, df: pd.DataFrame, text_column: str = None, 
                          sentiment_column: Optional[str] = None) -> pd.DataFrame:
        """
        Preprocess an entire dataset with filtering for short chunks.
        Auto-detects text/label columns if not provided.
        """
        # Auto-detect columns if not provided
        if text_column is None or (sentiment_column is None and sentiment_column != False):
            detected_text, detected_label = detect_text_and_label_columns(df)
            text_column = text_column or detected_text
            if sentiment_column is None:
                sentiment_column = detected_label
        if not text_column:
            raise ValueError("Could not detect text column in DataFrame.")
        processed_data = []
        total_filtered = 0
        for idx, row in df.iterrows():
            article_text = row[text_column]
            processed = self.preprocess_article(article_text)
            total_filtered += processed.get('filtered_short_chunks', 0)
            for chunk_idx, chunk in enumerate(processed['chunks']):
                entry = {
                    'original_id': idx,
                    'chunk_id': chunk_idx,
                    'text': chunk,
                    'is_long': processed['is_long'],
                    'total_chunks': len(processed['chunks']),
                    'num_tokens': processed['num_tokens'],
                    'cleaning_metrics': processed['cleaning_metrics']
                }
                # Add sentiment if available
                if sentiment_column and sentiment_column in row:
                    entry['sentiment'] = row[sentiment_column]
                # Add other columns
                for col in df.columns:
                    if col not in [text_column, sentiment_column]:
                        entry[col] = row[col]
                processed_data.append(entry)
        result_df = pd.DataFrame(processed_data)
        if hasattr(result_df, 'attrs'):
            result_df.attrs['total_filtered_chunks'] = total_filtered
            result_df.attrs['original_count'] = len(df)
            result_df.attrs['final_count'] = len(result_df)
        return result_df
    
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

class TweetPreprocessor:
    """Preprocessor for tweets for LSTM sentiment analysis."""
    def __init__(self, remove_stopwords=True):
        self.remove_stopwords = remove_stopwords
        nltk.download('stopwords', quiet=True)
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
        self.stemmer = PorterStemmer()

    def tweet_to_words(self, tweet):
        # Convert to lowercase
        text = tweet.lower()
        # Remove non-letters
        text = re.sub(r"[^a-zA-Z0-9]", " ", text)
        # Tokenize
        words = text.split()
        # Remove stopwords
        if self.remove_stopwords:
            words = [w for w in words if w not in self.stop_words]
        # Apply stemming
        words = [self.stemmer.stem(w) for w in words]
        return words

    def preprocess_tweets(self, tweets):
        return [self.tweet_to_words(t) for t in tweets]

class VocabBuilder:
    def __init__(self, max_vocab_size=5000):
        self.max_vocab_size = max_vocab_size
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = ['<PAD>', '<UNK>']

    def build_vocab(self, tokenized_texts):
        counter = Counter(w for tokens in tokenized_texts for w in tokens)
        most_common = counter.most_common(self.max_vocab_size - 2)
        for word, _ in most_common:
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)

    def encode(self, tokenized_texts, max_len=50):
        encoded = []
        for tokens in tokenized_texts:
            idxs = [self.word2idx.get(w, 1) for w in tokens]
            if len(idxs) < max_len:
                idxs += [0] * (max_len - len(idxs))
            else:
                idxs = idxs[:max_len]
            encoded.append(idxs)
        return torch.tensor(encoded, dtype=torch.long)
