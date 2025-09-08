"""
InvestIQ - AI-powered investment guide using scikit-learn regression models.

This package provides real-time investment guidance based on trained regression models,
incorporating user-specific trade history and market data for personalized recommendations.
"""

__version__ = "0.1.0"
__author__ = "InvestIQ Team"
__email__ = "contact@investiq.ai"

from .data.downloader import download_ticker_data
from .data.features import engineer_features
from .models.trainer import train_model
from .prediction.predictor import predict_return
from .backtesting.backtester import run_backtest

__all__ = [
    "download_ticker_data",
    "engineer_features", 
    "train_model",
    "predict_return",
    "run_backtest"
]
