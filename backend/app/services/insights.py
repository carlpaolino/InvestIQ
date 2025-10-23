"""Simple heuristic insight generator for the MVP stage."""

from __future__ import annotations

from datetime import datetime, timezone
from textwrap import shorten
from typing import Optional

from app.schemas import InsightResponse, MarketSummary


def _determine_sentiment(change_percent: float) -> str:
    if change_percent > 1.0:
        return "bullish"
    if change_percent < -1.0:
        return "bearish"
    return "neutral"


def _craft_headline(summary: MarketSummary) -> str:
    direction = "up" if summary.change >= 0 else "down"
    return f"{summary.symbol} {direction} {abs(summary.change_percent):.2f}% at ${summary.price:.2f}"


def _build_rationale(summary: MarketSummary, ocr_text: Optional[str]) -> str:
    pieces = [
        f"Last price ${summary.price:.2f} ({summary.change:+.2f}, {summary.change_percent:+.2f}%).",
        f"Session range ${summary.day_low:.2f} – ${summary.day_high:.2f}.",
    ]
    if summary.volume:
        pieces.append(f"Latest volume {summary.volume:,}.")
    if ocr_text:
        cleaned = " ".join(ocr_text.split())
        pieces.append(f"OCR context: {shorten(cleaned, width=160, placeholder='…')}")
    return " ".join(pieces)


def generate_insight(summary: MarketSummary, ocr_text: Optional[str]) -> InsightResponse:
    """Build a lightweight recommendation payload combining market metrics and screen context."""
    sentiment = _determine_sentiment(summary.change_percent)
    headline = _craft_headline(summary)
    rationale = _build_rationale(summary, ocr_text)

    return InsightResponse(
        ticker=summary.symbol,
        headline=headline,
        rationale=rationale,
        sentiment=sentiment,
        timestamp=datetime.now(timezone.utc),
        market=summary,
    )

