"""
Data processing module for cleaning and preparing data for model training.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any, Union
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


def process_raw_data(
    data: pd.DataFrame,
    ticker: str,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Process raw market data for model training.
    
    Args:
        data: Raw OHLCV data
        ticker: Stock ticker symbol
        save_path: Optional path to save processed data
        
    Returns:
        Processed DataFrame ready for feature engineering
    """
    if data.empty:
        logger.warning(f"Empty data provided for {ticker}")
        return pd.DataFrame()
    
    df = data.copy()
    
    # Ensure Date column is datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    elif df.index.name == 'Date' or isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['Date'], keep='last')
    
    # Handle missing values
    df = _handle_missing_values(df)
    
    # Add ticker column if not present
    if 'ticker' not in df.columns:
        df['ticker'] = ticker
    
    # Validate data quality
    df = _validate_data_quality(df, ticker)
    
    # Save processed data
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info(f"Processed data saved to {save_path}")
    
    logger.info(f"Processed {len(df)} records for {ticker}")
    return df


def load_user_data(
    file_path: str,
    data_type: str = "trades"
) -> pd.DataFrame:
    """
    Load user-specific data (trades, notes, etc.).
    
    Args:
        file_path: Path to user data file
        data_type: Type of data ('trades', 'notes', 'preferences')
        
    Returns:
        DataFrame with user data
    """
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.warning(f"User data file not found: {file_path}")
            return pd.DataFrame()
        
        # Load based on file extension
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix.lower() == '.json':
            df = pd.read_json(file_path)
        else:
            logger.error(f"Unsupported file format: {file_path.suffix}")
            return pd.DataFrame()
        
        # Process based on data type
        if data_type == "trades":
            df = _process_trade_data(df)
        elif data_type == "notes":
            df = _process_notes_data(df)
        elif data_type == "preferences":
            df = _process_preferences_data(df)
        
        logger.info(f"Loaded {len(df)} {data_type} records from {file_path}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading user data from {file_path}: {str(e)}")
        return pd.DataFrame()


def _handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in the dataset."""
    
    # Forward fill for price data
    price_cols = ['Open', 'High', 'Low', 'Close']
    for col in price_cols:
        if col in df.columns:
            df[col] = df[col].fillna(method='ffill')
    
    # Fill volume with 0 (no trading)
    if 'Volume' in df.columns:
        df['Volume'] = df['Volume'].fillna(0)
    
    # Drop rows where critical data is still missing
    critical_cols = ['Close']
    df = df.dropna(subset=critical_cols)
    
    return df


def _validate_data_quality(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Validate and clean data quality issues."""
    
    initial_len = len(df)
    
    # Remove rows with invalid prices
    price_cols = ['Open', 'High', 'Low', 'Close']
    for col in price_cols:
        if col in df.columns:
            df = df[df[col] > 0]  # Prices must be positive
    
    # Ensure High >= Low
    if 'High' in df.columns and 'Low' in df.columns:
        df = df[df['High'] >= df['Low']]
    
    # Ensure High >= Close and Low <= Close
    if all(col in df.columns for col in ['High', 'Low', 'Close']):
        df = df[(df['High'] >= df['Close']) & (df['Low'] <= df['Close'])]
    
    # Remove extreme outliers (prices that change by more than 50% in one day)
    if 'Close' in df.columns:
        daily_returns = df['Close'].pct_change()
        df = df[abs(daily_returns) <= 0.5]  # Remove 50%+ daily changes
    
    final_len = len(df)
    removed = initial_len - final_len
    
    if removed > 0:
        logger.warning(f"Removed {removed} invalid records for {ticker}")
    
    return df


def _process_trade_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process trade history data."""
    
    # Expected columns: date, ticker, action, quantity, price, notes
    required_cols = ['date', 'ticker', 'action']
    
    # Check for required columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.warning(f"Missing required columns in trade data: {missing_cols}")
        return df
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Standardize action values
    action_mapping = {
        'buy': 'buy', 'purchase': 'buy', 'long': 'buy',
        'sell': 'sell', 'short': 'sell', 'exit': 'sell',
        'hold': 'hold', 'wait': 'hold'
    }
    df['action'] = df['action'].str.lower().map(action_mapping).fillna('hold')
    
    # Add numeric action encoding
    action_encoding = {'buy': 1, 'hold': 0, 'sell': -1}
    df['action_encoded'] = df['action'].map(action_encoding)
    
    return df


def _process_notes_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process user notes data."""
    
    # Expected columns: date, ticker, note, sentiment, importance
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # Add sentiment encoding if not present
    if 'sentiment' in df.columns:
        sentiment_mapping = {
            'positive': 1, 'bullish': 1, 'good': 1,
            'negative': -1, 'bearish': -1, 'bad': -1,
            'neutral': 0, 'mixed': 0
        }
        df['sentiment_encoded'] = df['sentiment'].str.lower().map(sentiment_mapping).fillna(0)
    
    return df


def _process_preferences_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process user preferences data."""
    
    # Expected columns: preference_type, value, weight
    if 'weight' in df.columns:
        # Normalize weights to sum to 1
        df['weight'] = df['weight'] / df['weight'].sum()
    
    return df


def combine_market_and_user_data(
    market_data: pd.DataFrame,
    user_data: pd.DataFrame,
    user_data_type: str = "trades"
) -> pd.DataFrame:
    """
    Combine market data with user-specific data.
    
    Args:
        market_data: Processed market data
        user_data: User-specific data
        user_data_type: Type of user data
        
    Returns:
        Combined DataFrame
    """
    if market_data.empty or user_data.empty:
        logger.warning("Empty data provided for combination")
        return market_data
    
    # Ensure both have date columns
    if 'Date' not in market_data.columns or 'date' not in user_data.columns:
        logger.error("Missing date columns for data combination")
        return market_data
    
    # Convert dates to datetime
    market_data['Date'] = pd.to_datetime(market_data['Date'])
    user_data['date'] = pd.to_datetime(user_data['date'])
    
    # Merge user data with market data
    if user_data_type == "trades":
        # For trades, we might want to aggregate by date
        user_agg = user_data.groupby(['date', 'ticker']).agg({
            'action_encoded': 'mean',  # Average action sentiment
            'quantity': 'sum' if 'quantity' in user_data.columns else 'count'
        }).reset_index()
        
        combined = market_data.merge(
            user_agg,
            left_on=['Date', 'ticker'],
            right_on=['date', 'ticker'],
            how='left'
        )
        
        # Fill missing user data with neutral values
        combined['action_encoded'] = combined['action_encoded'].fillna(0)
        combined['quantity'] = combined['quantity'].fillna(0)
        
    else:
        # For other data types, simple merge
        combined = market_data.merge(
            user_data,
            left_on=['Date', 'ticker'],
            right_on=['date', 'ticker'],
            how='left'
        )
    
    # Remove duplicate date column
    if 'date' in combined.columns:
        combined = combined.drop('date', axis=1)
    
    logger.info(f"Combined market data with {user_data_type} data")
    return combined


def create_training_dataset(
    processed_data: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple:
    """
    Create training and testing datasets.
    
    Args:
        processed_data: Processed data with features
        feature_columns: List of feature column names
        target_column: Target variable column name
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    # Select features and target
    X = processed_data[feature_columns].copy()
    y = processed_data[target_column].copy()
    
    # Remove rows with missing values
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=None
    )
    
    logger.info(f"Created training dataset: {len(X_train)} train, {len(X_test)} test samples")
    
    return X_train, X_test, y_train, y_test
