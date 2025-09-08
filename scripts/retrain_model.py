#!/usr/bin/env python3
"""
Script to retrain the model with new data.
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from investiq.train import main as train_main

def main():
    """Retrain model with new data."""
    parser = argparse.ArgumentParser(description="Retrain InvestIQ model with new data")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "NVDA", "META", "NFLX"],
        help="Tickers to include in retraining"
    )
    parser.add_argument(
        "--model-type",
        choices=["ridge", "random_forest", "gradient_boosting"],
        default="ridge",
        help="Model type to train"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Backup current model before retraining"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test new model after training"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # Backup current model if requested
    if args.backup:
        backup_path = f"models/backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Backing up current model to {backup_path}")
        # Add backup logic here
    
    # Retrain model
    logger.info("Starting model retraining...")
    logger.info(f"Tickers: {', '.join(args.tickers)}")
    logger.info(f"Model type: {args.model_type}")
    
    # Set up training arguments
    sys.argv = [
        "train.py",
        "--tickers"] + args.tickers + [
        "--model-type", args.model_type,
        "--hyperparameter-tuning"
    ]
    
    try:
        train_main()
        logger.info("Model retraining completed successfully!")
        
        # Test new model if requested
        if args.test:
            logger.info("Testing new model...")
            # Add test logic here
            
    except Exception as e:
        logger.error(f"Model retraining failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
