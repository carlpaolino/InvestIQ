# InvestLens Overlay UI

Electron overlay that surfaces InvestLens insights on top of trading platforms. Connects to the FastAPI backend and renders market summaries, OCR output, and generated guidance.

## Features

- Frameless always-on-top window sized for desktop docking.
- Preload bridge exposing typed helper methods (`getMarketSummary`, `runOcr`, `generateInsight`).
- Renderer UI with ticker selector, OCR upload, and insight panel.

## Setup

1. Install dependencies:
   ```bash
   npm install
   ```
2. Ensure the backend is running on `http://127.0.0.1:8000` (default).
3. Launch the overlay:
   ```bash
   npm run start
   ```
   The window connects to the backend automatically. Set `BACKEND_URL` to point elsewhere:
   ```bash
   BACKEND_URL=http://192.168.1.20:8000 npm run start
   ```

## Testing Flow

1. Start the backend (`uvicorn app.main:app --reload`).
2. Start the overlay (`npm run start`).
3. Enter a ticker and fetch data to verify live market calls.
4. Drop a screenshot (PNG/JPG) into the OCR uploader to see extracted text feed the insight generator.

