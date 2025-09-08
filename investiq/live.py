#!/usr/bin/env python3
"""
Live prediction script for InvestIQ.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from investiq.prediction.predictor import predict_return

logger = logging.getLogger(__name__)


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description="Get live predictions from InvestIQ")
    parser.add_argument(
        "ticker",
        help="Stock ticker symbol (e.g., AAPL)"
    )
    parser.add_argument(
        "--model",
        help="Path to specific model (defaults to latest)"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="Prediction horizon in days (default: 5)"
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.6,
        help="Confidence threshold for suggestions (default: 0.6)"
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)"
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
        # Make prediction
        result = predict_return(
            ticker=args.ticker.upper(),
            model_path=args.model,
            horizon=args.horizon,
            confidence_threshold=args.confidence_threshold
        )
        
        # Format output
        if args.format == "json":
            import json
            output = json.dumps(result, indent=2, default=str)
        else:
            output = _format_result(result, args.horizon)
        
        print(output)
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        sys.exit(1)


def _format_result(result: dict, horizon: int) -> str:
    """Format prediction result for display."""
    if 'error' in result:
        return f"Error predicting {result['ticker']}: {result['error']}"
    
    output = []
    output.append(f"Ticker: {result['ticker']}")
    output.append(f"Predicted {horizon}D return: {result['prediction']:.4f} ({result['prediction']*100:.2f}%)")
    output.append(f"Suggestion: {result['suggestion']}")
    output.append(f"Confidence: {result['confidence']:.2f}")
    output.append(f"Model: {result['model_type']}")
    output.append(f"Training date: {result['training_date']}")
    
    if result.get('data_date'):
        output.append(f"Data date: {result['data_date']}")
    
    return "\n".join(output)


if __name__ == "__main__":
    main()
