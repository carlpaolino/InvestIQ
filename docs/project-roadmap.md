# InvestLens Roadmap

This roadmap adapts the provided outline into actionable phases for the InvestLens trading assistant. Each phase can be developed independently and merged once validated.

## Phase 0 — Environment
- Confirm Python 3.11+ and Node.js 20+ environments.
- Install core tooling: PyTorch, TensorFlow, transformers, OCR utilities, Electron.
- Define environment variables for API keys (Polygon, Alpha Vantage, etc.).

## Phase 1 — MVP Foundations
- **Screen capture + OCR**: Prototype with `mss` and `paddleocr`/`pytesseract`.
- **Market data adapters**: Wrap Yahoo Finance/Polygon endpoints with caching.
- **Backend API**: Deliver OCR text, price snapshots, and generated commentary.
- **Overlay UI**: Render backend messages, highlight key metrics, provide quick actions.

## Phase 2 — AI Core
- Train/evaluate PyTorch vision models for chart pattern recognition.
- Build TensorFlow time-series forecasters (LSTM/GRU) based on historical data.
- Fine-tune an LLM using trading notes and feedback loops.
- Harmonize outputs into a single recommendation schema for the UI.

## Phase 3 — Personalization & Feedback
- Capture user decisions and rationales for reinforcement learning pipelines.
- Iterate on model fine-tuning and prompt strategies.
- Add scenario testing/backtesting with frameworks like `backtrader`.

## Phase 4 — Deployment & Scaling
- Containerize services (Docker) and orchestrate components.
- Harden security (auth, sandboxing, secure storage of API keys).
- Package the overlay for desktop distribution (Electron builder/Tauri).

