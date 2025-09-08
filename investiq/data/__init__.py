"""
Data pipeline module for InvestIQ.

Handles data ingestion, processing, and feature engineering for market data
and user-specific trading information.
"""

from .downloader import download_ticker_data, download_multiple_tickers
from .features import engineer_features, create_target_variable
from .processor import process_raw_data, load_user_data

__all__ = [
    "download_ticker_data",
    "download_multiple_tickers", 
    "engineer_features",
    "create_target_variable",
    "process_raw_data",
    "load_user_data"
]
