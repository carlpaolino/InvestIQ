#!/usr/bin/env python3
"""
Backtesting script for InvestIQ strategies.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from investiq.backtesting.backtester import run_backtest, compare_strategies
from investiq.backtesting.strategy import SimpleStrategy, ThresholdStrategy, MomentumStrategy
from investiq.backtesting.metrics import generate_report, compare_strategies_report

logger = logging.getLogger(__name__)


def main():
    """Main backtesting function."""
    parser = argparse.ArgumentParser(description="Run backtests for InvestIQ strategies")
    parser.add_argument(
        "ticker",
        help="Stock ticker symbol to backtest"
    )
    parser.add_argument(
        "--model",
        help="Path to trained model (defaults to latest)"
    )
    parser.add_argument(
        "--strategy",
        choices=["simple", "threshold", "momentum", "all"],
        default="simple",
        help="Strategy to test (default: simple)"
    )
    parser.add_argument(
        "--start-date",
        help="Backtest start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        help="Backtest end date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=10000.0,
        help="Initial capital (default: 10000)"
    )
    parser.add_argument(
        "--transaction-cost",
        type=float,
        default=0.001,
        help="Transaction cost as fraction (default: 0.001)"
    )
    parser.add_argument(
        "--output",
        help="Output file for backtest report"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        if args.strategy == "all":
            # Test multiple strategies
            strategies = [
                SimpleStrategy(threshold=0.02),
                ThresholdStrategy(),
                MomentumStrategy()
            ]
            
            logger.info(f"Running backtest comparison for {args.ticker}")
            results = compare_strategies(
                ticker=args.ticker,
                model_path=args.model,
                strategies=strategies,
                start_date=args.start_date,
                end_date=args.end_date,
                initial_capital=args.initial_capital
            )
            
            # Generate comparison report
            report = compare_strategies_report(results)
            print(report)
            
        else:
            # Test single strategy
            if args.strategy == "simple":
                strategy = SimpleStrategy(threshold=0.02)
            elif args.strategy == "threshold":
                strategy = ThresholdStrategy()
            elif args.strategy == "momentum":
                strategy = MomentumStrategy()
            else:
                raise ValueError(f"Unknown strategy: {args.strategy}")
            
            logger.info(f"Running backtest for {args.ticker} with {args.strategy} strategy")
            result = run_backtest(
                ticker=args.ticker,
                model_path=args.model,
                strategy=strategy,
                start_date=args.start_date,
                end_date=args.end_date,
                initial_capital=args.initial_capital,
                transaction_cost=args.transaction_cost
            )
            
            # Generate report
            report = generate_report(result)
            print(report)
        
        # Save report if requested
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {args.output}")
        
    except Exception as e:
        logger.error(f"Backtest failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
