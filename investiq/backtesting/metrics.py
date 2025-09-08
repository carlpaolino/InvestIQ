"""
Performance metrics calculation for backtesting results.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def calculate_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.02
) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics.
    
    Args:
        strategy_returns: Strategy returns series
        benchmark_returns: Benchmark returns series
        risk_free_rate: Risk-free rate (annual)
        
    Returns:
        Dictionary of performance metrics
    """
    # Convert to numpy arrays for calculations
    strategy_returns = strategy_returns.dropna()
    benchmark_returns = benchmark_returns.dropna()
    
    # Align series
    common_index = strategy_returns.index.intersection(benchmark_returns.index)
    strategy_returns = strategy_returns.loc[common_index]
    benchmark_returns = benchmark_returns.loc[common_index]
    
    if len(strategy_returns) == 0:
        logger.warning("No overlapping returns data for metrics calculation")
        return {}
    
    # Basic return metrics
    total_return = (1 + strategy_returns).prod() - 1
    annualized_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
    
    benchmark_total_return = (1 + benchmark_returns).prod() - 1
    benchmark_annualized_return = (1 + benchmark_total_return) ** (252 / len(benchmark_returns)) - 1
    
    # Volatility metrics
    strategy_volatility = strategy_returns.std() * np.sqrt(252)
    benchmark_volatility = benchmark_returns.std() * np.sqrt(252)
    
    # Risk-adjusted metrics
    sharpe_ratio = (annualized_return - risk_free_rate) / strategy_volatility if strategy_volatility > 0 else 0
    benchmark_sharpe = (benchmark_annualized_return - risk_free_rate) / benchmark_volatility if benchmark_volatility > 0 else 0
    
    # Drawdown metrics
    strategy_cumulative = (1 + strategy_returns).cumprod()
    strategy_drawdown = _calculate_drawdown(strategy_cumulative)
    max_drawdown = strategy_drawdown.min()
    
    benchmark_cumulative = (1 + benchmark_returns).cumprod()
    benchmark_drawdown = _calculate_drawdown(benchmark_cumulative)
    benchmark_max_drawdown = benchmark_drawdown.min()
    
    # Win rate and other metrics
    win_rate = (strategy_returns > 0).mean()
    benchmark_win_rate = (benchmark_returns > 0).mean()
    
    # Information ratio
    excess_returns = strategy_returns - benchmark_returns
    information_ratio = excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
    
    # Calmar ratio
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Sortino ratio (downside deviation)
    downside_returns = strategy_returns[strategy_returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
    
    # Beta and correlation
    correlation = strategy_returns.corr(benchmark_returns)
    beta = correlation * strategy_volatility / benchmark_volatility if benchmark_volatility > 0 else 0
    
    # Alpha (Jensen's alpha)
    alpha = annualized_return - (risk_free_rate + beta * (benchmark_annualized_return - risk_free_rate))
    
    # Value at Risk (VaR) - 5% VaR
    var_95 = np.percentile(strategy_returns, 5)
    
    # Conditional Value at Risk (CVaR) - Expected shortfall
    cvar_95 = strategy_returns[strategy_returns <= var_95].mean()
    
    # Tail ratio
    tail_ratio = abs(np.percentile(strategy_returns, 5) / np.percentile(strategy_returns, 95)) if np.percentile(strategy_returns, 95) != 0 else 0
    
    metrics = {
        # Return metrics
        'total_return': total_return,
        'annualized_return': annualized_return,
        'benchmark_total_return': benchmark_total_return,
        'benchmark_annualized_return': benchmark_annualized_return,
        'excess_return': annualized_return - benchmark_annualized_return,
        
        # Risk metrics
        'volatility': strategy_volatility,
        'benchmark_volatility': benchmark_volatility,
        'max_drawdown': max_drawdown,
        'benchmark_max_drawdown': benchmark_max_drawdown,
        'var_95': var_95,
        'cvar_95': cvar_95,
        
        # Risk-adjusted metrics
        'sharpe_ratio': sharpe_ratio,
        'benchmark_sharpe': benchmark_sharpe,
        'information_ratio': information_ratio,
        'calmar_ratio': calmar_ratio,
        'sortino_ratio': sortino_ratio,
        'alpha': alpha,
        'beta': beta,
        
        # Other metrics
        'win_rate': win_rate,
        'benchmark_win_rate': benchmark_win_rate,
        'correlation': correlation,
        'tail_ratio': tail_ratio,
        'skewness': strategy_returns.skew(),
        'kurtosis': strategy_returns.kurtosis()
    }
    
    return metrics


def _calculate_drawdown(cumulative_returns: pd.Series) -> pd.Series:
    """Calculate drawdown series from cumulative returns."""
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    return drawdown


def generate_report(
    backtest_result: 'BacktestResult',
    include_trades: bool = True
) -> str:
    """
    Generate a comprehensive backtest report.
    
    Args:
        backtest_result: BacktestResult object
        include_trades: Whether to include trade details
        
    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 60)
    report.append("INVESTIQ BACKTEST REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Basic information
    report.append("BACKTEST PERIOD:")
    report.append(f"  Start Date: {backtest_result.start_date.strftime('%Y-%m-%d')}")
    report.append(f"  End Date: {backtest_result.end_date.strftime('%Y-%m-%d')}")
    report.append(f"  Initial Capital: ${backtest_result.initial_capital:,.2f}")
    report.append(f"  Final Capital: ${backtest_result.final_capital:,.2f}")
    report.append("")
    
    # Performance metrics
    metrics = backtest_result.metrics
    report.append("PERFORMANCE METRICS:")
    report.append(f"  Total Return: {metrics['total_return']:.2%}")
    report.append(f"  Annualized Return: {metrics['annualized_return']:.2%}")
    report.append(f"  Benchmark Return: {metrics['benchmark_annualized_return']:.2%}")
    report.append(f"  Excess Return: {metrics['excess_return']:.2%}")
    report.append("")
    
    # Risk metrics
    report.append("RISK METRICS:")
    report.append(f"  Volatility: {metrics['volatility']:.2%}")
    report.append(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
    report.append(f"  VaR (95%): {metrics['var_95']:.2%}")
    report.append(f"  CVaR (95%): {metrics['cvar_95']:.2%}")
    report.append("")
    
    # Risk-adjusted metrics
    report.append("RISK-ADJUSTED METRICS:")
    report.append(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    report.append(f"  Information Ratio: {metrics['information_ratio']:.2f}")
    report.append(f"  Calmar Ratio: {metrics['calmar_ratio']:.2f}")
    report.append(f"  Sortino Ratio: {metrics['sortino_ratio']:.2f}")
    report.append(f"  Alpha: {metrics['alpha']:.2%}")
    report.append(f"  Beta: {metrics['beta']:.2f}")
    report.append("")
    
    # Trade statistics
    report.append("TRADE STATISTICS:")
    report.append(f"  Total Trades: {backtest_result.total_trades}")
    report.append(f"  Win Rate: {metrics['win_rate']:.2%}")
    report.append(f"  Correlation with Benchmark: {metrics['correlation']:.2f}")
    report.append("")
    
    # Distribution metrics
    report.append("RETURN DISTRIBUTION:")
    report.append(f"  Skewness: {metrics['skewness']:.2f}")
    report.append(f"  Kurtosis: {metrics['kurtosis']:.2f}")
    report.append(f"  Tail Ratio: {metrics['tail_ratio']:.2f}")
    report.append("")
    
    # Trade details (if requested and available)
    if include_trades and not backtest_result.trades.empty:
        report.append("RECENT TRADES:")
        recent_trades = backtest_result.trades.tail(10)
        for _, trade in recent_trades.iterrows():
            report.append(f"  {trade['date'].strftime('%Y-%m-%d')}: "
                         f"Pred={trade['prediction']:.3f}, "
                         f"Signal={trade['signal']:.2f}, "
                         f"Return={trade['actual_return']:.2%}")
        report.append("")
    
    report.append("=" * 60)
    
    return "\n".join(report)


def compare_strategies_report(
    strategy_results: Dict[str, 'BacktestResult']
) -> str:
    """
    Generate comparison report for multiple strategies.
    
    Args:
        strategy_results: Dictionary mapping strategy names to BacktestResult objects
        
    Returns:
        Formatted comparison report
    """
    report = []
    report.append("=" * 80)
    report.append("STRATEGY COMPARISON REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Create comparison table
    report.append(f"{'Strategy':<20} {'Return':<10} {'Volatility':<12} {'Sharpe':<8} {'Max DD':<10} {'Win Rate':<10}")
    report.append("-" * 80)
    
    for strategy_name, result in strategy_results.items():
        if result is None:
            report.append(f"{strategy_name:<20} {'ERROR':<10} {'ERROR':<12} {'ERROR':<8} {'ERROR':<10} {'ERROR':<10}")
            continue
        
        metrics = result.metrics
        report.append(f"{strategy_name:<20} "
                     f"{metrics['annualized_return']:>8.2%} "
                     f"{metrics['volatility']:>10.2%} "
                     f"{metrics['sharpe_ratio']:>6.2f} "
                     f"{metrics['max_drawdown']:>8.2%} "
                     f"{metrics['win_rate']:>8.2%}")
    
    report.append("")
    report.append("=" * 80)
    
    return "\n".join(report)
