"""
Tests for tools.py — all network calls are mocked so no real API key needed.

Run from research_agent/ with:
    pytest tests/ -v
"""

import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tools import (
    get_company_news,
    get_stock_price,
    get_technical_info,
    search_tickers,
)


# ─── 1. get_company_news ─────────────────────────────────────────────────────

class TestGetCompanyNews:

    def test_missing_api_key_returns_error(self, monkeypatch):
        monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
        result = get_company_news("AAPL")
        assert "error" in result
        assert "ALPHA_VANTAGE_API_KEY" in result["error"]

    def test_successful_response_shape(self, monkeypatch):
        monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "fake_key")
        import requests

        fake_feed = {
            "feed": [{
                "title": "Apple hits record",
                "summary": "AAPL surged today.",
                "source": "Reuters",
                "time_published": "20260601T120000",
                "url": "https://reuters.com/aapl",
                "overall_sentiment_label": "Bullish",
                "overall_sentiment_score": "0.35",
                "ticker_sentiment": [{
                    "ticker": "AAPL",
                    "relevance_score": "0.9",
                    "ticker_sentiment_label": "Bullish",
                    "ticker_sentiment_score": "0.4",
                }],
            }]
        }

        class FakeResp:
            status_code = 200
            def raise_for_status(self): pass
            def json(self): return fake_feed

        monkeypatch.setattr(requests, "get", lambda *a, **kw: FakeResp())
        result = get_company_news("AAPL", limit=1)
        assert result["ticker"] == "AAPL"
        assert result["article_count"] == 1
        assert result["articles"][0]["title"] == "Apple hits record"

    def test_api_rate_limit_returns_error(self, monkeypatch):
        monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "fake_key")
        import requests

        class RateLimitResp:
            status_code = 200
            def raise_for_status(self): pass
            def json(self): return {"Note": "API rate limit reached."}

        monkeypatch.setattr(requests, "get", lambda *a, **kw: RateLimitResp())
        result = get_company_news("AAPL")
        assert "error" in result


# ─── 2. get_stock_price ──────────────────────────────────────────────────────

class TestGetStockPrice:

    def test_valid_ticker_returns_price(self, monkeypatch):
        import yfinance as yf

        class FakeTicker:
            info = {
                "currentPrice": 193.42,
                "previousClose": 192.10,
                "longName": "Apple Inc.",
                "exchange": "NMS",
                "sector": "Technology",
                "marketCap": 3_000_000_000_000,
                "trailingPE": 28.5,
                "targetMeanPrice": 220.0,
                "recommendationKey": "buy",
            }

        monkeypatch.setattr(yf, "Ticker", lambda t: FakeTicker())
        result = get_stock_price("AAPL")
        assert "error" not in result
        assert result["ticker"] == "AAPL"
        assert result["current_price"] == 193.42
        assert result["company_name"] == "Apple Inc."

    def test_change_percent_calculated_correctly(self, monkeypatch):
        import yfinance as yf

        class FakeTicker:
            info = {"currentPrice": 110.0, "previousClose": 100.0}

        monkeypatch.setattr(yf, "Ticker", lambda t: FakeTicker())
        result = get_stock_price("TEST")
        assert result["change_percent_today"] == pytest.approx(10.0, abs=0.01)

    def test_invalid_ticker_returns_error(self, monkeypatch):
        import yfinance as yf

        class FakeTicker:
            info = {}

        monkeypatch.setattr(yf, "Ticker", lambda t: FakeTicker())
        result = get_stock_price("FAKEXYZ")
        assert "error" in result

    def test_yfinance_exception_returns_error(self, monkeypatch):
        import yfinance as yf

        class BrokenTicker:
            @property
            def info(self):
                raise RuntimeError("network timeout")

        monkeypatch.setattr(yf, "Ticker", lambda t: BrokenTicker())
        result = get_stock_price("AAPL")
        assert "error" in result


# ─── 3. get_technical_info ───────────────────────────────────────────────────

class TestGetTechnicalInfo:

    def test_trend_signal_strong_uptrend(self, monkeypatch):
        import yfinance as yf

        class FakeTicker:
            info = {
                "currentPrice": 200.0,
                "fiftyDayAverage": 180.0,
                "twoHundredDayAverage": 160.0,
            }

        monkeypatch.setattr(yf, "Ticker", lambda t: FakeTicker())
        result = get_technical_info("AAPL")
        assert result["trend_signal"] == "strong_uptrend"

    def test_trend_signal_strong_downtrend(self, monkeypatch):
        import yfinance as yf

        class FakeTicker:
            info = {
                "currentPrice": 100.0,
                "fiftyDayAverage": 120.0,
                "twoHundredDayAverage": 150.0,
            }

        monkeypatch.setattr(yf, "Ticker", lambda t: FakeTicker())
        result = get_technical_info("AAPL")
        assert result["trend_signal"] == "strong_downtrend"

    def test_pct_above_ma50_calculation(self, monkeypatch):
        import yfinance as yf

        class FakeTicker:
            info = {
                "currentPrice": 110.0,
                "fiftyDayAverage": 100.0,
                "twoHundredDayAverage": 90.0,
            }

        monkeypatch.setattr(yf, "Ticker", lambda t: FakeTicker())
        result = get_technical_info("AAPL")
        assert result["pct_above_ma50"] == pytest.approx(10.0, abs=0.01)


# ─── 4. search_tickers ───────────────────────────────────────────────────────

class TestSearchTickers:

    def test_successful_search_shape(self, monkeypatch):
        import yfinance as yf

        class FakeSearch:
            quotes = [{"symbol": "AAPL", "longname": "Apple Inc.",
                       "exchange": "NMS", "quoteType": "EQUITY", "sector": None}]
            def __init__(self, *a, **kw): pass

        monkeypatch.setattr(yf, "Search", FakeSearch)
        result = search_tickers("Apple", limit=3)
        assert result["count"] == 1
        assert result["results"][0]["ticker"] == "AAPL"

    def test_empty_results_returns_zero_count(self, monkeypatch):
        import yfinance as yf

        class FakeSearch:
            quotes = []
            def __init__(self, *a, **kw): pass

        monkeypatch.setattr(yf, "Search", FakeSearch)
        result = search_tickers("unknownxyz")
        assert result["count"] == 0

    def test_yfinance_exception_returns_error(self, monkeypatch):
        import yfinance as yf

        def broken_search(*a, **kw):
            raise RuntimeError("connection refused")

        monkeypatch.setattr(yf, "Search", broken_search)
        result = search_tickers("Apple")
        assert "error" in result
