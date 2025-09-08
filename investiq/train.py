#!/usr/bin/env python3
"""
Training script for InvestIQ models.
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from investiq.data.downloader import download_multiple_tickers
from investiq.data.features import engineer_features
from investiq.data.processor import process_raw_data, combine_market_and_user_data, create_training_dataset
from investiq.models.trainer import train_model, save_model

logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train InvestIQ models")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "NVDA", "META", "NFLX"],
        help="Tickers to use for training"
    )
    parser.add_argument(
        "--model-type",
        choices=["ridge", "random_forest", "gradient_boosting"],
        default="ridge",
        help="Model type to train"
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds"
    )
    parser.add_argument(
        "--hyperparameter-tuning",
        action="store_true",
        help="Enable hyperparameter tuning"
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Output directory for trained models"
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Data directory"
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
    
    logger.info("Starting InvestIQ model training")
    logger.info(f"Tickers: {', '.join(args.tickers)}")
    logger.info(f"Model type: {args.model_type}")
    
    try:
        # Download and process data
        logger.info("Downloading market data...")
        raw_data_dir = Path(args.data_dir) / "raw"
        raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Download data for all tickers
        market_data = {}
        for ticker in args.tickers:
            logger.info(f"Downloading {ticker}...")
            data = download_multiple_tickers(
                [ticker],
                period="2y",
                save_dir=str(raw_data_dir)
            )
            if ticker in data and not data[ticker].empty:
                market_data[ticker] = data[ticker]
            else:
                logger.warning(f"Failed to download data for {ticker}")
        
        if not market_data:
            raise ValueError("No market data downloaded")
        
        # Process and engineer features
        logger.info("Processing data and engineering features...")
        all_features = []
        
        for ticker, data in market_data.items():
            logger.info(f"Processing {ticker}...")
            
            # Process raw data
            processed_data = process_raw_data(data, ticker)
            
            # Engineer features
            features_df = engineer_features(processed_data, target_horizon=5)
            
            if not features_df.empty:
                all_features.append(features_df)
                logger.info(f"Generated {len(features_df)} feature rows for {ticker}")
            else:
                logger.warning(f"No features generated for {ticker}")
        
        if not all_features:
            raise ValueError("No features generated from any ticker")
        
        # Combine all features
        combined_features = pd.concat(all_features, ignore_index=True)
        logger.info(f"Combined dataset: {len(combined_features)} rows, {len(combined_features.columns)} columns")
        
        # Prepare training data
        logger.info("Preparing training data...")
        
        # Select feature columns (exclude target and metadata columns)
        exclude_cols = ['Date', 'ticker', 'forward_return_5d', 'target_5d', 'target_class_5d']
        feature_columns = [col for col in combined_features.columns 
                          if col not in exclude_cols and pd.api.types.is_numeric_dtype(combined_features[col])]
        
        # Remove rows with missing target
        combined_features = combined_features.dropna(subset=['forward_return_5d'])
        
        # Create training dataset
        X_train, X_test, y_train, y_test = create_training_dataset(
            combined_features,
            feature_columns,
            'forward_return_5d',
            test_size=0.2
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        logger.info(f"Features: {len(feature_columns)}")
        
        # Train model
        logger.info(f"Training {args.model_type} model...")
        training_result = train_model(
            X_train=X_train,
            y_train=y_train,
            model_type=args.model_type,
            cv_folds=args.cv_folds,
            hyperparameter_tuning=args.hyperparameter_tuning
        )
        
        # Save model
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = output_dir / "latest"
        save_model(
            training_result['model'],
            training_result['metadata'],
            str(model_path)
        )
        
        # Print results
        metrics = training_result['metrics']
        logger.info("Training completed successfully!")
        logger.info(f"R² Score: {metrics['r2']:.4f}")
        logger.info(f"CV R² Score: {metrics['cv_r2_mean']:.4f}")
        logger.info(f"CV MSE: {metrics['cv_mse_mean']:.4f}")
        logger.info(f"Model saved to: {model_path}")
        
        # Feature importance (if available)
        if training_result['metadata']['feature_importance']:
            logger.info("Top 10 most important features:")
            feature_importance = training_result['metadata']['feature_importance']
            for i, (feature, importance) in enumerate(list(feature_importance.items())[:10]):
                logger.info(f"  {i+1:2d}. {feature}: {importance:.4f}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
