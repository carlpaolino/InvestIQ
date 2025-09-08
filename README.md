# InvestIQ

An AI-powered investment guide that uses scikit-learn regression models trained on market data and user-specific trading information to provide real-time investment recommendations.

## Overview

InvestIQ is a personal investment assistant that combines machine learning with financial data to generate actionable investment insights. The system uses regression models to predict future returns and provides clear buy/hold/sell recommendations based on your personal trading style and market conditions.

### Key Features

- **Real-time Predictions**: Get 5-day return predictions for any stock ticker
- **Personalized Models**: Incorporate your trading history and preferences
- **Multiple Algorithms**: Ridge regression, Random Forest, and Gradient Boosting
- **Comprehensive Backtesting**: Walk-forward backtesting with multiple strategies
- **Simple Interface**: CLI, API, and web interface options
- **Transparent Results**: Clear explanations with confidence scores

## Quick Start

### 1. Setup

Follow the detailed setup guide in [SETUP.md](SETUP.md) for complete installation instructions.

```bash
# Clone and navigate to project
cd "/Users/carl/Documents/Personal/CS Projects/InvestIQ"

# Create virtual environment
python3 -m venv investiq_env
source investiq_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create directory structure
mkdir -p data/{raw,processed} models logs
```

### 2. Download Sample Data

```bash
python scripts/download_sample_data.py
```

### 3. Train Your First Model

```bash
python investiq/train.py
```

### 4. Get Predictions

```bash
# Single ticker prediction
python investiq/live.py AAPL

# Batch predictions
python investiq/live.py AAPL MSFT GOOGL --batch
```

### 5. Run Backtests

```bash
python investiq/backtest.py AAPL --strategy simple
```

## Usage Examples

### Command Line Interface

```bash
# Basic prediction
python investiq/live.py AAPL

# Custom horizon and confidence
python investiq/live.py AAPL --horizon 10 --confidence-threshold 0.7

# JSON output
python investiq/live.py AAPL --format json

# Batch predictions
python investiq/live.py AAPL MSFT GOOGL --batch --output results.json
```

### API Interface

```bash
# Start API server
python investiq/prediction/api.py

# Access web interface
open http://localhost:8000/ui

# API endpoints
curl http://localhost:8000/score/AAPL
curl -X POST http://localhost:8000/batch -H "Content-Type: application/json" -d '["AAPL", "MSFT", "GOOGL"]'
```

### Training Custom Models

```bash
# Train with specific tickers
python investiq/train.py --tickers AAPL MSFT GOOGL --model-type random_forest

# Enable hyperparameter tuning
python investiq/train.py --hyperparameter-tuning

# Train multiple models
python investiq/train.py --model-type ridge --cv-folds 10
```

### Backtesting Strategies

```bash
# Simple strategy
python investiq/backtest.py AAPL --strategy simple

# Compare all strategies
python investiq/backtest.py AAPL --strategy all

# Custom date range
python investiq/backtest.py AAPL --start-date 2023-01-01 --end-date 2023-12-31
```

## Architecture

### Data Pipeline
- **Raw Data**: Yahoo Finance market data via yfinance
- **Feature Engineering**: 50+ technical indicators (RSI, MACD, Bollinger Bands, etc.)
- **User Data**: Trade history, notes, and preferences integration
- **Processing**: Data cleaning, validation, and target variable creation

### Model Training
- **Algorithms**: Ridge, Random Forest, HistGradientBoosting
- **Cross-Validation**: Time-series CV to prevent data leakage
- **Feature Selection**: Automatic feature importance ranking
- **Hyperparameter Tuning**: Grid search optimization

### Prediction Engine
- **Real-time Data**: Live market data fetching
- **Feature Computation**: On-the-fly technical indicator calculation
- **Confidence Scoring**: Prediction reliability assessment
- **Suggestion Generation**: Buy/hold/sell recommendations with thresholds

### Backtesting Framework
- **Walk-Forward**: Realistic out-of-sample testing
- **Multiple Strategies**: Simple, threshold, momentum, adaptive
- **Performance Metrics**: Sharpe ratio, max drawdown, win rate, alpha
- **Transaction Costs**: Realistic trading cost modeling

## Configuration

Edit `config/config.yaml` to customize:

- **Model Parameters**: Algorithms, CV folds, hyperparameter tuning
- **Feature Engineering**: Technical indicators, lookback periods
- **Prediction Settings**: Confidence thresholds, suggestion criteria
- **API Configuration**: Host, port, CORS settings
- **Data Sources**: Yahoo Finance settings, timeouts, retries

## Data Requirements

### Market Data
- **Source**: Yahoo Finance (yfinance)
- **Frequency**: Daily OHLCV data
- **History**: 2+ years recommended for training
- **Coverage**: US stocks, ETFs, indices

### User Data (Optional)
- **Trade History**: CSV with date, ticker, action, quantity, price
- **Notes**: JSON with date, ticker, sentiment, importance
- **Preferences**: Risk tolerance, sector preferences, time horizons

## Performance Metrics

### Model Performance
- **R² Score**: Explained variance in returns
- **Cross-Validation**: Out-of-sample performance
- **Feature Importance**: Most predictive indicators
- **Direction Accuracy**: Correct up/down predictions

### Strategy Performance
- **Total Return**: Cumulative performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Worst peak-to-trough loss
- **Win Rate**: Percentage of profitable trades
- **Information Ratio**: Active return vs tracking error

## Future Enhancements

### Planned Features
- **Floating Overlay UI**: Electron/Tauri desktop app
- **Explainable AI**: SHAP feature importance breakdown
- **Multi-Horizon Predictions**: 1D, 5D, 10D, 30D forecasts
- **Sentiment Integration**: News and social media analysis
- **Risk Management**: Position sizing and portfolio optimization
- **Automated Retraining**: Scheduled model updates

### Advanced Models
- **Deep Learning**: LSTM, Transformer architectures
- **Ensemble Methods**: Stacking and blending
- **Alternative Data**: Options flow, insider trading, earnings
- **Multi-Asset**: Cross-asset correlation modeling

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational and research purposes only. It is not intended as financial advice. Always do your own research and consult with qualified financial advisors before making investment decisions. Past performance does not guarantee future results.

## Support

- **Documentation**: See [SETUP.md](SETUP.md) for detailed setup instructions
- **Issues**: Report bugs and request features on GitHub
- **Discussions**: Join community discussions for questions and ideas

---

**InvestIQ** - Making investment decisions smarter with AI.