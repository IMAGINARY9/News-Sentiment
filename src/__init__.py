"""
News Sentiment Analysis Package

This package provides tools for sentiment analysis of news articles,
including preprocessing, model training, and evaluation utilities.
"""

__version__ = "0.1.0"
__author__ = "Sentiment Analysis Team"

from .preprocessing import NewsPreprocessor
from .models import NewsTransformerModel, HierarchicalModel
from .training import NewsTrainer
from .evaluation import NewsEvaluator

__all__ = [
    "NewsPreprocessor",
    "NewsTransformerModel", 
    "HierarchicalModel",
    "NewsTrainer",
    "NewsEvaluator"
]
