"""
Walk-forward backtesting engine for strategy evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Callable
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass

from ..data.downloader import download_ticker_data
from ..data.features import engineer_features
from ..models.trainer import load_model
from .strategy import SimpleStrategy
from .metrics import calculate_metrics
from ..prediction.predictor import _get_latest_model_path

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for backtest results."""
    strategy_returns: pd.Series
    benchmark_returns: pd.Series
    trades: pd.DataFrame
    metrics: Dict[str, float]
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    total_trades: int


def run_backtest(
    ticker: str,
    model_path: str,
    strategy: Optional[SimpleStrategy] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    initial_capital: float = 10000.0,
    transaction_cost: float = 0.001,
    rebalance_frequency: str = "daily"
) -> BacktestResult:
    """
    Run walk-forward backtest for a given strategy.
    
    Args:
        ticker: Stock ticker symbol
        model_path: Path to trained model
        strategy: Trading strategy (defaults to SimpleStrategy)
        start_date: Backtest start date (YYYY-MM-DD)
        end_date: Backtest end date (YYYY-MM-DD)
        initial_capital: Starting capital
        transaction_cost: Transaction cost as fraction (0.001 = 0.1%)
        rebalance_frequency: Rebalancing frequency ('daily', 'weekly', 'monthly')
        
    Returns:
        BacktestResult object with performance metrics
    """
    logger.info(f"Starting backtest for {ticker}")
    
    # Default strategy
    if strategy is None:
        strategy = SimpleStrategy(threshold=0.02)
    
    # Default date range (last 2 years)
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    
    # Download historical data
    data = download_ticker_data(
        ticker, 
        period="max",  # Get all available data
        interval="1d"
    )
    
    if data.empty:
        raise ValueError(f"No data available for {ticker}")
    
    # Filter by date range
    data['Date'] = pd.to_datetime(data['Date'])
    data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
    
    if len(data) < 50:  # Need minimum data for backtest
        raise ValueError(f"Insufficient data for backtest: {len(data)} days")
    
    # Engineer features
    features_df = engineer_features(data, target_horizon=5)
    if features_df.empty:
        raise ValueError("Could not engineer features for backtest")
    
    # Load model
    if model_path is None:
        model_path = _get_latest_model_path()
    
    model, metadata = load_model(model_path)
    
    # Run walk-forward backtest
    backtest_data = _run_walk_forward_backtest(
        features_df, model, strategy, metadata,
        initial_capital, transaction_cost, rebalance_frequency
    )
    
    # Calculate performance metrics
    metrics = calculate_metrics(
        backtest_data['strategy_returns'],
        backtest_data['benchmark_returns']
    )
    
    # Create result object
    result = BacktestResult(
        strategy_returns=backtest_data['strategy_returns'],
        benchmark_returns=backtest_data['benchmark_returns'],
        trades=backtest_data['trades'],
        metrics=metrics,
        start_date=pd.to_datetime(start_date),
        end_date=pd.to_datetime(end_date),
        initial_capital=initial_capital,
        final_capital=backtest_data['final_capital'],
        max_drawdown=metrics['max_drawdown'],
        sharpe_ratio=metrics['sharpe_ratio'],
        win_rate=metrics['win_rate'],
        total_trades=len(backtest_data['trades'])
    )
    
    logger.info(f"Backtest complete. Final capital: ${result.final_capital:.2f}")
    logger.info(f"Total return: {metrics['total_return']:.2%}")
    logger.info(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
    
    return result


def _run_walk_forward_backtest(
    features_df: pd.DataFrame,
    model: Any,
    strategy: SimpleStrategy,
    metadata: Dict[str, Any],
    initial_capital: float,
    transaction_cost: float,
    rebalance_frequency: str
) -> Dict[str, Any]:
    """
    Run walk-forward backtest simulation.
    
    Args:
        features_df: DataFrame with features and targets
        model: Trained model
        strategy: Trading strategy
        metadata: Model metadata
        initial_capital: Starting capital
        transaction_cost: Transaction cost
        rebalance_frequency: Rebalancing frequency
        
    Returns:
        Dictionary with backtest results
    """
    # Initialize tracking variables
    capital = initial_capital
    position = 0.0  # Fraction of capital invested
    trades = []
    strategy_returns = []
    benchmark_returns = []
    
    # Get feature columns
    feature_columns = metadata.get('feature_columns', [])
    if not feature_columns:
        # Fallback: use all numeric columns except targets
        exclude_cols = ['Date', 'ticker', 'forward_return_5d', 'target_5d', 'target_class_5d']
        feature_columns = [col for col in features_df.columns 
                         if col not in exclude_cols and pd.api.types.is_numeric_dtype(features_df[col])]
    
    # Sort by date
    features_df = features_df.sort_values('Date').reset_index(drop=True)
    
    # Determine rebalancing schedule
    rebalance_dates = _get_rebalance_dates(features_df['Date'], rebalance_frequency)
    
    for i, date in enumerate(rebalance_dates):
        if i == 0:
            continue  # Skip first date (no prediction available)
        
        # Get data up to current date (walk-forward)
        current_data = features_df[features_df['Date'] <= date].copy()
        
        if len(current_data) < 30:  # Need minimum data for prediction
            continue
        
        # Get latest features
        latest_features = current_data.iloc[-1:][feature_columns].copy()
        latest_features = latest_features.fillna(latest_features.median())
        
        # Make prediction
        try:
            prediction = model.predict(latest_features)[0]
        except Exception as e:
            logger.warning(f"Prediction failed for {date}: {str(e)}")
            continue
        
        # Get actual return for this period
        if i < len(rebalance_dates) - 1:
            next_date = rebalance_dates[i + 1]
            current_price = current_data['Close'].iloc[-1]
            
            # Find next price
            next_data = features_df[features_df['Date'] <= next_date]
            if len(next_data) > len(current_data):
                next_price = next_data['Close'].iloc[-1]
                actual_return = (next_price - current_price) / current_price
            else:
                continue
        else:
            continue
        
        # Generate signal from strategy
        signal = strategy.generate_signal(prediction)
        
        # Execute trade
        new_position = signal
        position_change = new_position - position
        
        # Calculate transaction costs
        transaction_cost_amount = abs(position_change) * capital * transaction_cost
        
        # Update capital
        capital = capital - transaction_cost_amount
        
        # Calculate strategy return
        if position != 0:
            strategy_return = position * actual_return
        else:
            strategy_return = 0.0
        
        # Update capital with strategy return
        capital = capital * (1 + strategy_return)
        
        # Record trade
        if abs(position_change) > 0.01:  # Only record significant position changes
            trades.append({
                'date': date,
                'prediction': prediction,
                'signal': signal,
                'position': new_position,
                'position_change': position_change,
                'actual_return': actual_return,
                'strategy_return': strategy_return,
                'capital': capital,
                'transaction_cost': transaction_cost_amount
            })
        
        # Update position
        position = new_position
        
        # Record returns
        strategy_returns.append(strategy_return)
        benchmark_returns.append(actual_return)
    
    # Create returns series
    strategy_returns_series = pd.Series(strategy_returns, index=rebalance_dates[1:len(strategy_returns)+1])
    benchmark_returns_series = pd.Series(benchmark_returns, index=rebalance_dates[1:len(benchmark_returns)+1])
    
    # Create trades DataFrame
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    
    return {
        'strategy_returns': strategy_returns_series,
        'benchmark_returns': benchmark_returns_series,
        'trades': trades_df,
        'final_capital': capital
    }


def _get_rebalance_dates(dates: pd.Series, frequency: str) -> List[datetime]:
    """Get rebalancing dates based on frequency."""
    dates = pd.to_datetime(dates).sort_values()
    
    if frequency == "daily":
        return dates.tolist()
    elif frequency == "weekly":
        # Get first trading day of each week
        weekly_dates = dates.groupby(dates.dt.isocalendar().week).first()
        return weekly_dates.tolist()
    elif frequency == "monthly":
        # Get first trading day of each month
        monthly_dates = dates.groupby(dates.dt.to_period('M')).first()
        return monthly_dates.tolist()
    else:
        raise ValueError(f"Unknown rebalancing frequency: {frequency}")


def compare_strategies(
    ticker: str,
    model_path: str,
    strategies: List[SimpleStrategy],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    initial_capital: float = 10000.0
) -> Dict[str, BacktestResult]:
    """
    Compare multiple strategies on the same data.
    
    Args:
        ticker: Stock ticker symbol
        model_path: Path to trained model
        strategies: List of strategies to compare
        start_date: Backtest start date
        end_date: Backtest end date
        initial_capital: Starting capital
        
    Returns:
        Dictionary mapping strategy names to BacktestResult objects
    """
    results = {}
    
    for strategy in strategies:
        strategy_name = strategy.__class__.__name__
        logger.info(f"Backtesting strategy: {strategy_name}")
        
        try:
            result = run_backtest(
                ticker=ticker,
                model_path=model_path,
                strategy=strategy,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital
            )
            results[strategy_name] = result
        except Exception as e:
            logger.error(f"Failed to backtest {strategy_name}: {str(e)}")
            results[strategy_name] = None
    
    return results
