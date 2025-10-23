"""Pydantic models shared across InvestLens backend endpoints."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., example="ok")
    message: str = Field(..., example="InvestLens backend ready")


class MarketSummary(BaseModel):
    symbol: str = Field(..., example="AAPL")
    price: float = Field(..., example=175.42)
    change: float = Field(..., description="Absolute price change since previous close")
    change_percent: float = Field(..., description="Percent change since previous close")
    day_high: float = Field(..., example=178.1)
    day_low: float = Field(..., example=172.3)
    volume: Optional[int] = Field(None, description="Latest volume figure when available")
    timestamp: datetime = Field(..., description="Timestamp of the latest price quote")


class MarketSummaryResponse(BaseModel):
    data: MarketSummary


class OcrResult(BaseModel):
    text: str = Field(..., description="Extracted text from the supplied image")
    confidence: Optional[float] = Field(
        None, description="Average OCR confidence when available"
    )


class InsightRequest(BaseModel):
    ticker: str = Field(..., example="AAPL")
    ocr_text: Optional[str] = Field(
        None, description="Optional OCR text to provide additional context"
    )


class InsightResponse(BaseModel):
    ticker: str
    headline: str
    rationale: str
    sentiment: str
    timestamp: datetime
    market: MarketSummary

