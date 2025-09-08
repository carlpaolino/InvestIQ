"""
Backtesting module for InvestIQ.

Provides walk-forward backtesting and strategy evaluation capabilities.
"""

from .backtester import run_backtest, BacktestResult
from .strategy import SimpleStrategy, ThresholdStrategy
from .metrics import calculate_metrics, generate_report

__all__ = [
    "run_backtest",
    "BacktestResult",
    "SimpleStrategy", 
    "ThresholdStrategy",
    "calculate_metrics",
    "generate_report"
]
