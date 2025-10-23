"""Utilities for retrieving market data from public APIs."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import yfinance as yf

from app.schemas import MarketSummary


class MarketDataError(RuntimeError):
    """Raised when market data could not be retrieved or parsed."""


def _extract_latest_snapshot(history: pd.DataFrame) -> tuple[datetime, float, float]:
    if history.empty:
        raise MarketDataError("No intraday quotes available for the requested symbol.")
    latest_row = history.iloc[-1]
    timestamp = latest_row.name.to_pydatetime()
    price = float(latest_row["Close"])
    volume = float(latest_row.get("Volume", 0.0))
    return timestamp, price, volume


def fetch_summary(symbol: str) -> MarketSummary:
    """Fetch a lightweight market summary for the provided ticker symbol."""
    ticker = yf.Ticker(symbol)

    intraday = ticker.history(period="1d", interval="1m")
    timestamp, price, volume = _extract_latest_snapshot(intraday)

    daily = ticker.history(period="5d", interval="1d")
    if daily.empty:
        raise MarketDataError("Unable to compute previous close for ticker.")

    # Use the last completed trading day's close as the baseline when possible.
    if len(daily) >= 2:
        previous_close = float(daily["Close"].iloc[-2])
    else:
        previous_close = float(daily["Close"].iloc[-1])
    change = price - previous_close
    change_percent = (change / previous_close) * 100 if previous_close else 0.0

    # Day high/low from the current session when available.
    session_high = float(intraday["High"].max()) if "High" in intraday else price
    session_low = float(intraday["Low"].min()) if "Low" in intraday else price

    return MarketSummary(
        symbol=symbol.upper(),
        price=round(price, 4),
        change=round(change, 4),
        change_percent=round(change_percent, 2),
        day_high=round(session_high, 4),
        day_low=round(session_low, 4),
        volume=int(volume) if volume else None,
        timestamp=timestamp,
    )
