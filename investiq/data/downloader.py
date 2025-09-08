"""
Data downloader module for fetching market data from various sources.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
import logging
from pathlib import Path
import time

logger = logging.getLogger(__name__)


def download_ticker_data(
    ticker: str,
    period: str = "2y",
    interval: str = "1d",
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Download historical data for a single ticker.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        save_path: Optional path to save the data as CSV
        
    Returns:
        DataFrame with OHLCV data and additional columns
    """
    try:
        logger.info(f"Downloading data for {ticker} (period: {period}, interval: {interval})")
        
        # Create yfinance ticker object
        stock = yf.Ticker(ticker)
        
        # Download historical data
        data = stock.history(period=period, interval=interval)
        
        if data.empty:
            logger.warning(f"No data found for ticker {ticker}")
            return pd.DataFrame()
        
        # Add ticker column
        data['ticker'] = ticker
        
        # Add additional metrics
        data = _add_additional_metrics(data, stock)
        
        # Reset index to make Date a column
        data = data.reset_index()
        
        # Save if path provided
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            data.to_csv(save_path, index=False)
            logger.info(f"Data saved to {save_path}")
        
        logger.info(f"Successfully downloaded {len(data)} records for {ticker}")
        return data
        
    except Exception as e:
        logger.error(f"Error downloading data for {ticker}: {str(e)}")
        return pd.DataFrame()


def download_multiple_tickers(
    tickers: List[str],
    period: str = "2y",
    interval: str = "1d",
    save_dir: Optional[str] = None,
    delay: float = 0.1
) -> Dict[str, pd.DataFrame]:
    """
    Download historical data for multiple tickers.
    
    Args:
        tickers: List of stock ticker symbols
        period: Data period
        interval: Data interval
        save_dir: Optional directory to save individual CSV files
        delay: Delay between downloads to avoid rate limiting
        
    Returns:
        Dictionary mapping ticker symbols to DataFrames
    """
    results = {}
    
    for ticker in tickers:
        try:
            save_path = None
            if save_dir:
                save_path = Path(save_dir) / f"{ticker}.csv"
            
            data = download_ticker_data(ticker, period, interval, save_path)
            results[ticker] = data
            
            # Add delay to avoid rate limiting
            if delay > 0:
                time.sleep(delay)
                
        except Exception as e:
            logger.error(f"Failed to download {ticker}: {str(e)}")
            results[ticker] = pd.DataFrame()
    
    return results


def _add_additional_metrics(data: pd.DataFrame, stock: yf.Ticker) -> pd.DataFrame:
    """
    Add additional metrics to the downloaded data.
    
    Args:
        data: OHLCV DataFrame
        stock: yfinance Ticker object
        
    Returns:
        DataFrame with additional metrics
    """
    try:
        # Get additional info
        info = stock.info
        
        # Add market cap if available
        if 'marketCap' in info:
            data['market_cap'] = info['marketCap']
        
        # Add sector and industry if available
        if 'sector' in info:
            data['sector'] = info['sector']
        if 'industry' in info:
            data['industry'] = info['industry']
            
        # Add dividend yield if available
        if 'dividendYield' in info:
            data['dividend_yield'] = info['dividendYield']
            
    except Exception as e:
        logger.warning(f"Could not fetch additional metrics: {str(e)}")
    
    return data


def get_ticker_info(ticker: str) -> Dict[str, Any]:
    """
    Get detailed information about a ticker.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary with ticker information
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Extract key information
        key_info = {
            'symbol': ticker,
            'name': info.get('longName', ''),
            'sector': info.get('sector', ''),
            'industry': info.get('industry', ''),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'dividend_yield': info.get('dividendYield', 0),
            'beta': info.get('beta', 0),
            '52_week_high': info.get('fiftyTwoWeekHigh', 0),
            '52_week_low': info.get('fiftyTwoWeekLow', 0)
        }
        
        return key_info
        
    except Exception as e:
        logger.error(f"Error getting info for {ticker}: {str(e)}")
        return {'symbol': ticker, 'error': str(e)}


def validate_ticker(ticker: str) -> bool:
    """
    Validate if a ticker symbol exists and has data.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        True if ticker is valid, False otherwise
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="5d")
        return not data.empty
    except Exception:
        return False
