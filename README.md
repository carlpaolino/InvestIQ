# InvestLens

InvestLens is an in-progress desktop overlay that watches the market with you, pulls real-time data, and offers AI-assisted trade notes. The current build ships a working FastAPI backend plus an Electron UI so you can exercise the end-to-end loop while we iterate on richer models.

## What's Implemented
- Live market summaries powered by Yahoo Finance (`/market/summary`).
- OCR endpoint (`/ocr/extract`) using pytesseract for screenshot text extraction.
- Insight generator (`/insights/generate`) that blends market deltas with optional OCR context.
- Electron overlay with ticker selector, OCR uploader, and rendered insights.

## Prerequisites
- **Python 3.11+**
- **Node.js 20+**
- **Tesseract OCR**
  - macOS: `brew install tesseract`
  - Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
  - Windows: install from <https://github.com/UB-Mannheim/tesseract/wiki>
  - Verify with `tesseract --version`

## Setup
1. **Backend**
   ```bash
   cd backend
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
   ```
   The API will be available at `http://127.0.0.1:8000`.

2. **Frontend**
   ```bash
   cd frontend
   npm install
   npm run start
   ```
   The overlay launches in a frameless window. Drag it wherever you like; it stays on top by default.

## Testing the Loop
1. With both backend and frontend running, confirm the status badge in the overlay reads “Backend online”.
2. Enter a ticker (e.g., `AAPL`) and press **Fetch Market Summary** to see live pricing.
3. Capture a portion of your screen (PNG/JPG) and drop it into the OCR uploader—the extracted text appears in the textarea.
4. Click **Generate Insight** to combine the OCR context with the most recent market snapshot.

## API Keys and External Data
- Yahoo Finance via `yfinance` does not require keys and ships enabled out of the box.
- If you plan to add Polygon.io, Alpha Vantage, or news feeds, create a `.env` file in `backend/` and add:
  ```env
  ALPHA_VANTAGE_KEY=your-key
  POLYGON_KEY=your-key
  NEWS_API_KEY=your-key
  ```
  Update the backend services to read these values (see `docs/project-roadmap.md` for planned integrations).

## Automated Tests
Run the FastAPI unit tests (network requests are mocked):
```bash
cd backend
pytest
```

## Project Structure
- `backend/`: FastAPI service, OCR/market/insight helpers, pytest suite.
- `frontend/`: Electron app with preload bridge and renderer assets.
- `docs/`: Roadmap, setup notes, and design artifacts.

## Next Steps
1. Expand the backend with real-time websocket streaming and richer ML inference.
2. Train domain models (vision + sequence) once curated datasets are available.
3. Harden the overlay UX (multi-panel layouts, notifications, user feedback loop).
4. Review “Cluely” demos for interaction patterns you want to mirror or improve.

