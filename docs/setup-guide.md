# InvestLens Local Setup Guide

Follow these steps to run the full stack locally and prepare for future integrations (market APIs, personal datasets, and databases).

## 1. System Dependencies
1. Install Python 3.11+ and Node.js 20+ (use pyenv/nvm if you manage multiple versions).
2. Install Tesseract OCR:
   - macOS: `brew install tesseract`
   - Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
   - Windows: download the MSI linked on the [Tesseract wiki](https://github.com/UB-Mannheim/tesseract/wiki).
3. Optional (recommended for later phases):
   - `brew install redis` (macOS) or `sudo apt-get install redis` to support caching/offline queues.
   - `brew install ffmpeg` / `sudo apt-get install ffmpeg` for future voice/video capture.

## 2. Python Backend
1. Create a virtualenv in `backend/`:
   ```bash
   cd backend
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
2. (Optional) Create `backend/.env` for API keys:
   ```env
   ALPHA_VANTAGE_KEY=your-key
   POLYGON_KEY=your-key
   NEWS_API_KEY=your-key
   ```
   The current build only uses Yahoo Finance, but environment variables are ready for upcoming adapters.
3. Start the server:
   ```bash
   uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
   ```
4. Run the automated tests (mocks external calls):
   ```bash
   pytest
   ```

## 3. Node/Electron Frontend
1. Install dependencies:
   ```bash
   cd frontend
   npm install
   ```
2. Start the overlay:
   ```bash
   npm run start
   ```
   Configure a remote backend by exporting `BACKEND_URL` before `npm run start`.

## 4. Verifying End-to-End
1. Ensure the backend reports healthy: `curl http://127.0.0.1:8000/health`.
2. In the overlay, enter `AAPL` (or any symbol) and fetch the summary.
3. Drag an image of market headlines or a price table into the OCR uploader to see extracted text.
4. Generate an insight and confirm sentiment/headline updates.

## 5. Preparing Data for Model Training
- Save OCR results and model input data under `data/raw/` (create the directory when you are ready).
- Use `docs/project-roadmap.md` to plan collection of trade labels and feedback loops.
- When you're ready to train, add notebooks under `research/` (create as needed) and reference them from the docs.

## 6. Optional Database Layer
If you intend to persist user decisions:
1. Spin up PostgreSQL locally (`brew install postgresql` or `docker run postgres`).
2. Create a database (e.g., `investlens_dev`) and store credentials in `backend/.env`.
3. Add an ORM (SQLAlchemy) to `backend/requirements.txt` and scaffold under `app/db/`.

By following these steps you'll have a fully operational development environment with clear hooks for future data sources, persistence, and model training.

