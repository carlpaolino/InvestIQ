"""
Core prediction engine for generating investment recommendations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import joblib
import json

from ..data.downloader import download_ticker_data
from ..data.features import engineer_features
from ..models.trainer import load_model

logger = logging.getLogger(__name__)


def predict_return(
    ticker: str,
    model_path: Optional[str] = None,
    horizon: int = 5,
    confidence_threshold: float = 0.6
) -> Dict[str, Any]:
    """
    Predict future returns for a given ticker.
    
    Args:
        ticker: Stock ticker symbol
        model_path: Path to trained model (defaults to latest)
        horizon: Prediction horizon in days
        confidence_threshold: Minimum confidence for predictions
        
    Returns:
        Dictionary with prediction results
    """
    try:
        logger.info(f"Making prediction for {ticker} (horizon: {horizon} days)")
        
        # Load model
        if model_path is None:
            model_path = _get_latest_model_path()
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        model, metadata = load_model(model_path)
        
        # Download latest data (need more data for feature engineering)
        data = download_ticker_data(ticker, period="2y", interval="1d")
        if data.empty:
            raise ValueError(f"No data available for {ticker}")
        
        # Engineer features
        features_df = engineer_features(data, target_horizon=horizon)
        if features_df.empty:
            raise ValueError(f"Could not engineer features for {ticker}")
        
        # Get latest features (most recent row)
        latest_features = features_df.iloc[-1:].copy()
        
        # Select feature columns used in training
        feature_columns = metadata.get('feature_columns', [])
        if not feature_columns:
            # Fallback: use all numeric columns except target columns
            exclude_cols = ['Date', 'ticker', 'forward_return_5d', 'target_5d', 'target_class_5d']
            feature_columns = [col for col in latest_features.columns 
                             if col not in exclude_cols and pd.api.types.is_numeric_dtype(latest_features[col])]
        
        # Prepare features for prediction
        X_pred = latest_features[feature_columns].copy()
        
        # Handle missing values
        X_pred = X_pred.fillna(X_pred.median())
        
        # Make prediction
        prediction = model.predict(X_pred)[0]
        
        # Calculate confidence (simplified - could be enhanced with prediction intervals)
        confidence = _calculate_prediction_confidence(model, X_pred, prediction)
        
        # Generate suggestion
        suggestion = generate_suggestion(
            prediction, 
            confidence, 
            confidence_threshold,
            metadata.get('suggestion_thresholds', {})
        )
        
        result = {
            'ticker': ticker,
            'prediction': prediction,
            'confidence': confidence,
            'suggestion': suggestion,
            'horizon_days': horizon,
            'model_type': metadata.get('model_type', 'unknown'),
            'training_date': metadata.get('training_date', 'unknown'),
            'feature_count': len(feature_columns),
            'data_date': latest_features['Date'].iloc[0].isoformat() if 'Date' in latest_features.columns else None
        }
        
        logger.info(f"Prediction complete: {prediction:.4f} ({suggestion})")
        return result
        
    except Exception as e:
        logger.error(f"Error making prediction for {ticker}: {str(e)}")
        return {
            'ticker': ticker,
            'error': str(e),
            'prediction': None,
            'suggestion': 'error'
        }


def generate_suggestion(
    prediction: float,
    confidence: float,
    confidence_threshold: float = 0.6,
    custom_thresholds: Optional[Dict[str, float]] = None
) -> str:
    """
    Generate investment suggestion based on prediction and confidence.
    
    Args:
        prediction: Predicted return value
        confidence: Prediction confidence (0-1)
        custom_thresholds: Custom suggestion thresholds
        
    Returns:
        Investment suggestion string
    """
    # Default thresholds
    thresholds = {
        'strong_buy': 0.05,
        'buy': 0.02,
        'hold': -0.02,
        'sell': -0.05,
        'strong_sell': -0.1
    }
    
    # Use custom thresholds if provided
    if custom_thresholds:
        thresholds.update(custom_thresholds)
    
    # Check confidence first
    if confidence < confidence_threshold:
        return "hold (low confidence)"
    
    # Generate suggestion based on prediction
    if prediction >= thresholds['strong_buy']:
        return "strong buy"
    elif prediction >= thresholds['buy']:
        return "buy"
    elif prediction >= thresholds['hold']:
        return "hold"
    elif prediction >= thresholds['sell']:
        return "sell"
    else:
        return "strong sell"


def _get_latest_model_path() -> str:
    """Get path to the latest trained model."""
    models_dir = Path("models")
    if not models_dir.exists():
        raise FileNotFoundError("Models directory not found")
    
    # Look for latest.joblib
    latest_model = models_dir / "latest.joblib"
    if latest_model.exists():
        return str(latest_model)
    
    # Look for other model files
    model_files = list(models_dir.glob("*.joblib"))
    if not model_files:
        raise FileNotFoundError("No trained models found")
    
    # Return the most recent model
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    return str(latest_model)


def _calculate_prediction_confidence(
    model: Any,
    X_pred: pd.DataFrame,
    prediction: float
) -> float:
    """
    Calculate prediction confidence (simplified implementation).
    
    Args:
        model: Trained model
        X_pred: Prediction features
        prediction: Model prediction
        
    Returns:
        Confidence score (0-1)
    """
    try:
        # For tree-based models, we can use prediction variance
        if hasattr(model, 'estimators_'):
            # Random Forest - use prediction variance
            predictions = [est.predict(X_pred)[0] for est in model.estimators_]
            variance = np.var(predictions)
            confidence = 1 / (1 + variance)  # Higher variance = lower confidence
        elif hasattr(model, 'named_steps') and hasattr(model.named_steps['model'], 'estimators_'):
            # Pipeline with Random Forest
            predictions = [est.predict(X_pred)[0] for est in model.named_steps['model'].estimators_]
            variance = np.var(predictions)
            confidence = 1 / (1 + variance)
        else:
            # For linear models, use a simple heuristic based on feature magnitude
            feature_magnitude = np.mean(np.abs(X_pred.values))
            confidence = min(0.9, 0.5 + feature_magnitude * 0.1)
        
        return max(0.1, min(0.9, confidence))  # Clamp between 0.1 and 0.9
        
    except Exception as e:
        logger.warning(f"Could not calculate confidence: {str(e)}")
        return 0.5  # Default confidence


def batch_predict(
    tickers: list,
    model_path: Optional[str] = None,
    horizon: int = 5
) -> Dict[str, Dict[str, Any]]:
    """
    Make predictions for multiple tickers.
    
    Args:
        tickers: List of ticker symbols
        model_path: Path to trained model
        horizon: Prediction horizon in days
        
    Returns:
        Dictionary mapping tickers to prediction results
    """
    results = {}
    
    for ticker in tickers:
        try:
            result = predict_return(ticker, model_path, horizon)
            results[ticker] = result
        except Exception as e:
            logger.error(f"Failed to predict {ticker}: {str(e)}")
            results[ticker] = {
                'ticker': ticker,
                'error': str(e),
                'prediction': None,
                'suggestion': 'error'
            }
    
    return results


def get_prediction_summary(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate summary statistics from batch predictions.
    
    Args:
        results: Dictionary of prediction results
        
    Returns:
        Summary statistics
    """
    valid_predictions = [r for r in results.values() if 'error' not in r and r['prediction'] is not None]
    
    if not valid_predictions:
        return {'error': 'No valid predictions found'}
    
    predictions = [r['prediction'] for r in valid_predictions]
    suggestions = [r['suggestion'] for r in valid_predictions]
    
    # Count suggestions
    suggestion_counts = {}
    for suggestion in suggestions:
        suggestion_counts[suggestion] = suggestion_counts.get(suggestion, 0) + 1
    
    summary = {
        'total_tickers': len(results),
        'valid_predictions': len(valid_predictions),
        'avg_prediction': np.mean(predictions),
        'median_prediction': np.median(predictions),
        'std_prediction': np.std(predictions),
        'min_prediction': np.min(predictions),
        'max_prediction': np.max(predictions),
        'suggestion_distribution': suggestion_counts,
        'positive_predictions': sum(1 for p in predictions if p > 0),
        'negative_predictions': sum(1 for p in predictions if p < 0)
    }
    
    return summary
