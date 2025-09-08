# InvestIQ Setup Guide

This guide provides exact step-by-step instructions to set up InvestIQ, an AI-powered investment guide using scikit-learn regression models.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (for version control)

## Step-by-Step Setup

### 1. Clone and Navigate to Project
```bash
cd "/Users/carl/Documents/Personal/CS Projects/InvestIQ"
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python3 -m venv investiq_env

# Activate virtual environment
source investiq_env/bin/activate

# Verify activation (you should see (investiq_env) in your prompt)
which python
```

### 3. Install Dependencies
```bash
# Upgrade pip first
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

### 4. Create Directory Structure
```bash
# Create data directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models
mkdir -p logs

# Create source code directories
mkdir -p investiq/data
mkdir -p investiq/models
mkdir -p investiq/prediction
mkdir -p investiq/backtesting
mkdir -p investiq/api
```

### 5. Set Up Configuration
```bash
# Create config file
cp config/config.yaml.example config/config.yaml

# Edit config with your preferences (optional)
# nano config/config.yaml
```

### 6. Download Sample Data (Optional)
```bash
# Download sample market data for testing
python scripts/download_sample_data.py
```

### 7. Train Initial Model
```bash
# Train the model with sample data
python investiq/train.py

# This will create models/latest.joblib
```

### 8. Test the System
```bash
# Test CLI prediction
python investiq/live.py AAPL

# Test API (in another terminal)
python investiq/api/app.py
# Then visit http://localhost:8000/docs for API documentation
```

### 9. Run Backtest
```bash
# Run backtest to evaluate model performance
python investiq/backtest.py
```

## Verification Steps

### Check Installation
```bash
# Verify all packages are installed
pip list | grep -E "(scikit-learn|yfinance|fastapi|pandas|numpy)"

# Test imports
python -c "import sklearn, yfinance, fastapi, pandas, numpy; print('All imports successful')"
```

### Test Data Pipeline
```bash
# Test data download
python -c "from investiq.data.downloader import download_ticker_data; print(download_ticker_data('AAPL', '1mo'))"

# Test feature engineering
python -c "from investiq.data.features import engineer_features; print('Feature engineering works')"
```

### Test Model Training
```bash
# Quick model test
python -c "from investiq.models.trainer import train_model; print('Model training works')"
```

## Troubleshooting

### Common Issues

1. **Virtual Environment Not Activating**
   ```bash
   # On macOS/Linux
   source investiq_env/bin/activate
   
   # On Windows
   investiq_env\Scripts\activate
   ```

2. **Permission Errors**
   ```bash
   # Fix permissions
   chmod +x scripts/*.py
   chmod +x investiq/*.py
   ```

3. **Missing Dependencies**
   ```bash
   # Reinstall requirements
   pip install -r requirements.txt --force-reinstall
   ```

4. **Data Download Issues**
   ```bash
   # Check internet connection and try again
   python scripts/download_sample_data.py --verbose
   ```

### Environment Variables (Optional)
```bash
# Add to your shell profile (~/.zshrc or ~/.bashrc)
export INVESTIQ_DATA_PATH="/Users/carl/Documents/Personal/CS Projects/InvestIQ/data"
export INVESTIQ_MODEL_PATH="/Users/carl/Documents/Personal/CS Projects/InvestIQ/models"
```

## Development Setup

### For Development
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
python -m pytest tests/
```

### IDE Setup (VS Code)
1. Install Python extension
2. Select the virtual environment interpreter
3. Install recommended extensions:
   - Python
   - Pylance
   - Jupyter

## Next Steps

After successful setup:

1. **Customize Configuration**: Edit `config/config.yaml` for your preferences
2. **Add Your Data**: Place your trade history in `data/raw/` as CSV files
3. **Train Custom Model**: Run `python investiq/train.py` with your data
4. **Test Predictions**: Use `python investiq/live.py TICKER` for predictions
5. **Evaluate Performance**: Run `python investiq/backtest.py` for strategy evaluation

## Support

If you encounter issues:
1. Check the logs in `logs/` directory
2. Verify all dependencies are installed correctly
3. Ensure you're using the correct Python version (3.8+)
4. Check that all directory permissions are correct

## Uninstallation

To remove InvestIQ:
```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf investiq_env

# Remove project files (optional)
# rm -rf "/Users/carl/Documents/Personal/CS Projects/InvestIQ"
```
