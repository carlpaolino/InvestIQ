# InvestLens Backend

FastAPI service that powers OCR extraction, market data retrieval, and heuristic insight generation for the InvestLens desktop overlay.

## Features

- `/health`: Service heartbeat.
- `/market/summary`: Retrieves live quotes and daily stats using Yahoo Finance via `yfinance`.
- `/ocr/extract`: Runs pytesseract on uploaded screenshots (FormData `file`).
- `/insights/generate`: Merges market data with optional OCR context to deliver a quick recommendation.

## Setup

1. **Python environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
2. **Install Tesseract (required for OCR)**
   - macOS (Homebrew): `brew install tesseract`
   - Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
   - Windows: installer at <https://github.com/UB-Mannheim/tesseract/wiki>
   - Confirm availability: `tesseract --version`
3. **Run the API**
   ```bash
   uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
   ```

## Testing

Tests stub external calls so you can run them without live market access:

```bash
pytest
```

## Example requests

```bash
# Health check
curl http://127.0.0.1:8000/health

# Market summary
curl "http://127.0.0.1:8000/market/summary?ticker=AAPL"

# Insight generation (OCR text optional)
curl -X POST http://127.0.0.1:8000/insights/generate \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "ocr_text": "Watching Apple breakout above resistance."}'
```

