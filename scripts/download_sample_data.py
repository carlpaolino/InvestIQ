#!/usr/bin/env python3
"""
Script to download sample data for testing and development.
"""

import argparse
import logging
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from investiq.data.downloader import download_multiple_tickers

logger = logging.getLogger(__name__)


def main():
    """Download sample data for testing."""
    parser = argparse.ArgumentParser(description="Download sample market data")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"],
        help="Tickers to download (default: AAPL MSFT GOOGL TSLA AMZN)"
    )
    parser.add_argument(
        "--period",
        default="2y",
        help="Data period (default: 2y)"
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Output directory (default: data/raw)"
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
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading data for {len(args.tickers)} tickers")
    logger.info(f"Tickers: {', '.join(args.tickers)}")
    logger.info(f"Period: {args.period}")
    logger.info(f"Output directory: {output_dir}")
    
    # Download data
    results = download_multiple_tickers(
        tickers=args.tickers,
        period=args.period,
        save_dir=str(output_dir),
        delay=0.1
    )
    
    # Report results
    successful = sum(1 for data in results.values() if not data.empty)
    failed = len(results) - successful
    
    logger.info(f"Download complete: {successful} successful, {failed} failed")
    
    for ticker, data in results.items():
        if data.empty:
            logger.warning(f"Failed to download {ticker}")
        else:
            logger.info(f"Downloaded {len(data)} records for {ticker}")


if __name__ == "__main__":
    main()
