from datetime import datetime, timezone
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import InsightResponse, MarketSummary


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def sample_summary() -> MarketSummary:
    return MarketSummary(
        symbol="TEST",
        price=110.0,
        change=10.0,
        change_percent=10.0,
        day_high=112.0,
        day_low=98.0,
        volume=123456,
        timestamp=datetime.now(timezone.utc),
    )


def test_healthcheck(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"


def test_generate_insight(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    def fake_fetch_summary(_: str) -> MarketSummary:
        return sample_summary()

    monkeypatch.setattr("app.main.fetch_summary", fake_fetch_summary)

    response = client.post("/insights/generate", json={"ticker": "TEST"})
    assert response.status_code == 200
    data = InsightResponse(**response.json())
    assert data.ticker == "TEST"
    assert data.sentiment == "bullish"


def test_market_summary(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    def fake_fetch_summary(_: str) -> MarketSummary:
        return sample_summary()

    monkeypatch.setattr("app.main.fetch_summary", fake_fetch_summary)

    response = client.get("/market/summary", params={"ticker": "TEST"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == "TEST"
    assert payload["price"] == pytest.approx(110.0)

