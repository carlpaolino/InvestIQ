"""
Command-line interface for InvestIQ predictions.
"""

import argparse
import sys
import logging
from typing import List, Optional
import json

from .predictor import predict_return, batch_predict, get_prediction_summary

logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="InvestIQ - AI-powered investment predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m investiq.prediction.cli AAPL
  python -m investiq.prediction.cli AAPL MSFT GOOGL --horizon 10
  python -m investiq.prediction.cli --batch AAPL MSFT GOOGL --output results.json
        """
    )
    
    # Positional arguments
    parser.add_argument(
        'tickers',
        nargs='*',
        help='Stock ticker symbols to analyze'
    )
    
    # Optional arguments
    parser.add_argument(
        '--model',
        type=str,
        help='Path to trained model (defaults to latest)'
    )
    
    parser.add_argument(
        '--horizon',
        type=int,
        default=5,
        help='Prediction horizon in days (default: 5)'
    )
    
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.6,
        help='Minimum confidence threshold for suggestions (default: 0.6)'
    )
    
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process multiple tickers in batch mode'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output file for batch results (JSON format)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--format',
        choices=['text', 'json'],
        default='text',
        help='Output format (default: text)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Validate arguments
    if not args.tickers:
        parser.error("At least one ticker symbol is required")
    
    try:
        if args.batch or len(args.tickers) > 1:
            # Batch mode
            results = batch_predict(
                args.tickers,
                model_path=args.model,
                horizon=args.horizon
            )
            
            if args.format == 'json':
                output = json.dumps(results, indent=2, default=str)
            else:
                output = _format_batch_results(results, args.horizon)
            
            print(output)
            
            # Save to file if requested
            if args.output:
                with open(args.output, 'w') as f:
                    if args.format == 'json':
                        json.dump(results, f, indent=2, default=str)
                    else:
                        f.write(output)
                print(f"Results saved to {args.output}")
        
        else:
            # Single ticker mode
            result = predict_return(
                args.tickers[0],
                model_path=args.model,
                horizon=args.horizon,
                confidence_threshold=args.confidence_threshold
            )
            
            if args.format == 'json':
                output = json.dumps(result, indent=2, default=str)
            else:
                output = _format_single_result(result, args.horizon)
            
            print(output)
            
            # Save to file if requested
            if args.output:
                with open(args.output, 'w') as f:
                    if args.format == 'json':
                        json.dump(result, f, indent=2, default=str)
                    else:
                        f.write(output)
                print(f"Result saved to {args.output}")
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)


def _format_single_result(result: dict, horizon: int) -> str:
    """Format single prediction result for display."""
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


def _format_batch_results(results: dict, horizon: int) -> str:
    """Format batch prediction results for display."""
    output = []
    output.append(f"InvestIQ Batch Predictions ({horizon}D horizon)")
    output.append("=" * 50)
    
    # Summary
    summary = get_prediction_summary(results)
    if 'error' not in summary:
        output.append(f"Total tickers: {summary['total_tickers']}")
        output.append(f"Valid predictions: {summary['valid_predictions']}")
        output.append(f"Average prediction: {summary['avg_prediction']:.4f}")
        output.append(f"Positive predictions: {summary['positive_predictions']}")
        output.append(f"Negative predictions: {summary['negative_predictions']}")
        output.append("")
    
    # Individual results
    for ticker, result in results.items():
        if 'error' in result:
            output.append(f"{ticker}: ERROR - {result['error']}")
        else:
            pred_pct = result['prediction'] * 100
            output.append(f"{ticker}: {result['prediction']:.4f} ({pred_pct:+.2f}%) - {result['suggestion']}")
    
    return "\n".join(output)


if __name__ == "__main__":
    main()
