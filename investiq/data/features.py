"""
Feature engineering module for creating technical indicators and derived features.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
import logging
from scipy import stats

logger = logging.getLogger(__name__)


def engineer_features(
    data: pd.DataFrame,
    lookback_days: int = 252,
    target_horizon: int = 5
) -> pd.DataFrame:
    """
    Engineer technical features from OHLCV data.
    
    Args:
        data: DataFrame with OHLCV data
        lookback_days: Number of days to look back for calculations
        target_horizon: Number of days forward for target variable
        
    Returns:
        DataFrame with engineered features
    """
    if data.empty:
        logger.warning("Empty data provided for feature engineering")
        return pd.DataFrame()
    
    # Make a copy to avoid modifying original data
    df = data.copy()
    
    # Ensure we have required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return pd.DataFrame()
    
    # Sort by date if not already sorted
    if 'Date' in df.columns:
        df = df.sort_values('Date').reset_index(drop=True)
    
    logger.info(f"Engineering features for {len(df)} records")
    
    # Price-based features
    df = _add_price_features(df)
    
    # Volume features
    df = _add_volume_features(df)
    
    # Technical indicators
    df = _add_technical_indicators(df)
    
    # Moving averages and gaps
    df = _add_moving_averages(df)
    
    # Volatility features
    df = _add_volatility_features(df, lookback_days)
    
    # Momentum features
    df = _add_momentum_features(df)
    
    # Time-based features
    df = _add_time_features(df)
    
    # Market regime features
    df = _add_market_regime_features(df)
    
    # Create target variable
    df = create_target_variable(df, target_horizon)
    
    # Remove rows with NaN values (due to lookback calculations)
    initial_len = len(df)
    df = df.dropna()
    final_len = len(df)
    
    logger.info(f"Feature engineering complete. {initial_len - final_len} rows removed due to NaN values")
    
    # Ensure we have at least one row
    if len(df) == 0:
        logger.warning("No valid rows after feature engineering")
        return pd.DataFrame()
    
    return df


def _add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add basic price-based features."""
    
    # Price ratios
    df['high_low_ratio'] = df['High'] / df['Low']
    df['close_open_ratio'] = df['Close'] / df['Open']
    df['high_close_ratio'] = df['High'] / df['Close']
    df['low_close_ratio'] = df['Low'] / df['Close']
    
    # Price position within daily range
    df['close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
    
    # Body and shadow ratios (candlestick patterns)
    df['body_ratio'] = abs(df['Close'] - df['Open']) / (df['High'] - df['Low'])
    df['upper_shadow'] = (df['High'] - np.maximum(df['Open'], df['Close'])) / (df['High'] - df['Low'])
    df['lower_shadow'] = (np.minimum(df['Open'], df['Close']) - df['Low']) / (df['High'] - df['Low'])
    
    return df


def _add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume-based features."""
    
    # Volume moving averages
    df['volume_ma_5'] = df['Volume'].rolling(5).mean()
    df['volume_ma_20'] = df['Volume'].rolling(20).mean()
    
    # Volume ratios
    df['volume_ratio_5'] = df['Volume'] / df['volume_ma_5']
    df['volume_ratio_20'] = df['Volume'] / df['volume_ma_20']
    
    # Volume-weighted average price (VWAP)
    df['vwap'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).rolling(20).sum() / df['Volume'].rolling(20).sum()
    df['price_vwap_ratio'] = df['Close'] / df['vwap']
    
    return df


def _add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators."""
    
    # RSI (Relative Strength Index)
    df['rsi_14'] = _calculate_rsi(df['Close'], 14)
    df['rsi_30'] = _calculate_rsi(df['Close'], 30)
    
    # MACD
    macd_line, signal_line, histogram = _calculate_macd(df['Close'])
    df['macd'] = macd_line
    df['macd_signal'] = signal_line
    df['macd_histogram'] = histogram
    
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = _calculate_bollinger_bands(df['Close'], 20, 2)
    df['bb_upper'] = bb_upper
    df['bb_middle'] = bb_middle
    df['bb_lower'] = bb_lower
    df['bb_width'] = (bb_upper - bb_lower) / bb_middle
    df['bb_position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
    
    # Stochastic Oscillator
    df['stoch_k'], df['stoch_d'] = _calculate_stochastic(df['High'], df['Low'], df['Close'])
    
    return df


def _add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Add moving averages and gaps."""
    
    # Simple Moving Averages
    for period in [5, 10, 20, 50, 100, 200]:
        df[f'sma_{period}'] = df['Close'].rolling(period).mean()
        df[f'price_sma_{period}_ratio'] = df['Close'] / df[f'sma_{period}']
    
    # Exponential Moving Averages
    for period in [12, 26, 50]:
        df[f'ema_{period}'] = df['Close'].ewm(span=period).mean()
        df[f'price_ema_{period}_ratio'] = df['Close'] / df[f'ema_{period}']
    
    # Moving average crossovers
    df['sma_5_20_cross'] = (df['sma_5'] > df['sma_20']).astype(int)
    df['sma_20_50_cross'] = (df['sma_20'] > df['sma_50']).astype(int)
    df['ema_12_26_cross'] = (df['ema_12'] > df['ema_26']).astype(int)
    
    return df


def _add_volatility_features(df: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    """Add volatility-based features."""
    
    # Daily returns
    df['daily_return'] = df['Close'].pct_change()
    
    # Rolling volatility
    for period in [5, 10, 20, 30]:
        df[f'volatility_{period}'] = df['daily_return'].rolling(period).std()
        df[f'volatility_{period}_annualized'] = df[f'volatility_{period}'] * np.sqrt(252)
    
    # Average True Range (ATR)
    df['atr_14'] = _calculate_atr(df['High'], df['Low'], df['Close'], 14)
    df['atr_ratio'] = df['atr_14'] / df['Close']
    
    # Volatility percentiles
    df['volatility_percentile_20'] = df['volatility_20'].rolling(lookback_days).rank(pct=True)
    
    return df


def _add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add momentum-based features."""
    
    # Price momentum
    for period in [1, 5, 10, 20, 50]:
        df[f'momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1
    
    # Rate of Change (ROC)
    for period in [5, 10, 20]:
        df[f'roc_{period}'] = df['Close'].pct_change(period)
    
    # Commodity Channel Index (CCI)
    df['cci_20'] = _calculate_cci(df['High'], df['Low'], df['Close'], 20)
    
    return df


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features."""
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Day of week (0=Monday, 6=Sunday)
        df['day_of_week'] = df['Date'].dt.dayofweek
        
        # Month
        df['month'] = df['Date'].dt.month
        
        # Quarter
        df['quarter'] = df['Date'].dt.quarter
        
        # Day of month
        df['day_of_month'] = df['Date'].dt.day
        
        # Is month end
        df['is_month_end'] = df['Date'].dt.is_month_end.astype(int)
        
        # Is quarter end
        df['is_quarter_end'] = df['Date'].dt.is_quarter_end.astype(int)
        
        # Days since last trading day
        df['days_since_last'] = df['Date'].diff().dt.days
    
    return df


def _add_market_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add market regime features."""
    
    # Trend strength
    df['trend_strength_20'] = (df['Close'] - df['Close'].shift(20)) / df['Close'].shift(20)
    df['trend_strength_50'] = (df['Close'] - df['Close'].shift(50)) / df['Close'].shift(50)
    
    # Market regime classification
    df['bull_market'] = (df['trend_strength_50'] > 0.1).astype(int)
    df['bear_market'] = (df['trend_strength_50'] < -0.1).astype(int)
    df['sideways_market'] = ((df['trend_strength_50'] >= -0.1) & (df['trend_strength_50'] <= 0.1)).astype(int)
    
    return df


def create_target_variable(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Create target variable for prediction.
    
    Args:
        df: DataFrame with price data
        horizon: Number of days forward to predict
        
    Returns:
        DataFrame with target variable added
    """
    # Forward returns
    df[f'forward_return_{horizon}d'] = df['Close'].shift(-horizon) / df['Close'] - 1
    
    # Binary classification target (positive/negative return)
    df[f'target_{horizon}d'] = (df[f'forward_return_{horizon}d'] > 0).astype(int)
    
    # Multi-class target (strong buy, buy, hold, sell, strong sell)
    df[f'target_class_{horizon}d'] = pd.cut(
        df[f'forward_return_{horizon}d'],
        bins=[-np.inf, -0.05, -0.02, 0.02, 0.05, np.inf],
        labels=[0, 1, 2, 3, 4]  # 0=strong_sell, 1=sell, 2=hold, 3=buy, 4=strong_buy
    ).astype(float)
    
    return df


# Technical indicator calculation functions

def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """Calculate MACD indicator."""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2) -> tuple:
    """Calculate Bollinger Bands."""
    sma = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band


def _calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> tuple:
    """Calculate Stochastic Oscillator."""
    lowest_low = low.rolling(period).min()
    highest_high = high.rolling(period).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(3).mean()
    return k_percent, d_percent


def _calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(period).mean()
    return atr


def _calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """Calculate Commodity Channel Index."""
    typical_price = (high + low + close) / 3
    sma_tp = typical_price.rolling(period).mean()
    mad = typical_price.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
    cci = (typical_price - sma_tp) / (0.015 * mad)
    return cci
