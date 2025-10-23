"""Entry point for the InvestLens backend service.

The backend currently supports:
- OCR extraction from uploaded screenshots (pytesseract)
- Market summaries powered by Yahoo Finance (via yfinance)
- Heuristic insight generation combining OCR context with market data

These endpoints are enough to exercise the overlay UI while we iterate on more
advanced model integrations.
"""

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import (
    HealthResponse,
    InsightRequest,
    InsightResponse,
    MarketSummary,
    OcrResult,
)
from app.services.insights import generate_insight
from app.services.market import MarketDataError, fetch_summary
from app.services.ocr import OcrServiceError, extract_text


app = FastAPI(title="InvestLens Backend", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Dev-only convenience; tighten for production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    """Confirm the service is reachable."""
    return HealthResponse(status="ok", message="InvestLens backend ready")


@app.get(
    "/market/summary",
    response_model=MarketSummary,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Ticker not found"},
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "Market data temporarily unavailable"
        },
    },
)
def market_summary(ticker: str) -> MarketSummary:
    """Return latest price movement details for a ticker symbol."""
    try:
        return fetch_summary(ticker)
    except MarketDataError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error while fetching market data: {exc}",
        ) from exc


@app.post(
    "/ocr/extract",
    response_model=OcrResult,
    responses={
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid image payload"},
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "OCR dependencies not ready"
        },
    },
)
async def ocr_extract(file: UploadFile = File(...)) -> OcrResult:
    """Run OCR on an uploaded screenshot."""
    try:
        payload = await file.read()
        return extract_text(payload)
    except OcrServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process image: {exc}",
        ) from exc


@app.post(
    "/insights/generate",
    response_model=InsightResponse,
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "Upstream data temporarily unavailable"
        }
    },
)
def generate_trading_insight(payload: InsightRequest) -> InsightResponse:
    """Produce a quick investment insight combining OCR context and market data."""
    try:
        summary = fetch_summary(payload.ticker)
    except MarketDataError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error while fetching market data: {exc}",
        ) from exc

    return generate_insight(summary, payload.ocr_text)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
