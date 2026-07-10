"""
AstraQuant — Lightweight Vercel Serverless API
Handles basic endpoints without heavy dependencies (PyTorch, LangGraph).
The full research/prediction backend must run locally.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

app = FastAPI(title="AstraQuant — Vercel API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=2)

# ─── Trending Tickers (static, no dependencies) ──────────────────────────────
TRENDING_TICKERS = [
    {"ticker": "NVDA", "name": "NVIDIA Corporation", "sector": "Semiconductors"},
    {"ticker": "AAPL", "name": "Apple Inc.", "sector": "Technology"},
    {"ticker": "MSFT", "name": "Microsoft Corp.", "sector": "Technology"},
    {"ticker": "GOOGL", "name": "Alphabet Inc.", "sector": "Technology"},
    {"ticker": "META", "name": "Meta Platforms", "sector": "Technology"},
    {"ticker": "AMZN", "name": "Amazon.com Inc.", "sector": "E-Commerce"},
    {"ticker": "TSLA", "name": "Tesla Inc.", "sector": "Automotive / EV"},
    {"ticker": "AMD", "name": "Advanced Micro Devices", "sector": "Semiconductors"},
    {"ticker": "INTC", "name": "Intel Corporation", "sector": "Semiconductors"},
    {"ticker": "QCOM", "name": "Qualcomm Inc.", "sector": "Semiconductors"},
    {"ticker": "ARM", "name": "Arm Holdings", "sector": "Semiconductors"},
    {"ticker": "SMCI", "name": "Super Micro Computer", "sector": "AI / Servers"},
    {"ticker": "PLTR", "name": "Palantir Technologies", "sector": "AI / Data"},
    {"ticker": "JPM", "name": "JPMorgan Chase", "sector": "Finance"},
    {"ticker": "GS", "name": "Goldman Sachs Group", "sector": "Finance"},
    {"ticker": "NFLX", "name": "Netflix Inc.", "sector": "Streaming"},
    {"ticker": "COIN", "name": "Coinbase Global", "sector": "Crypto"},
    {"ticker": "SPOT", "name": "Spotify Technology", "sector": "Music / Media"},
    {"ticker": "UBER", "name": "Uber Technologies", "sector": "Mobility"},
    {"ticker": "SHOP", "name": "Shopify Inc.", "sector": "E-Commerce"},
    # Pakistan
    {"ticker": "HBL.KA", "name": "Habib Bank Limited", "sector": "Pakistan / Banking"},
    {"ticker": "LUCK.KA", "name": "Lucky Cement Ltd.", "sector": "Pakistan / Cement"},
    {"ticker": "ENGRO.KA", "name": "Engro Corporation", "sector": "Pakistan / Conglomerate"},
    {"ticker": "PSO.KA", "name": "Pakistan State Oil", "sector": "Pakistan / Energy"},
    {"ticker": "OGDC.KA", "name": "Oil & Gas Dev. Co.", "sector": "Pakistan / Energy"},
    {"ticker": "MCB.KA", "name": "MCB Bank Limited", "sector": "Pakistan / Banking"},
    {"ticker": "SYS.KA", "name": "Systems Limited", "sector": "Pakistan / IT"},
]


# ─── Request Models ──────────────────────────────────────────────────────────
class ResearchRequest(BaseModel):
    ticker: str
    limit: Optional[int] = 8

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Ticker is required.")
        return v


class CompareRequest(BaseModel):
    ticker_a: str
    ticker_b: str


class WatchlistRequest(BaseModel):
    tickers: List[str]


# ─── Health ───────────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "AstraQuant Vercel API", "version": "3.0.0"}


# ─── Trending ─────────────────────────────────────────────────────────────────
@app.get("/api/trending")
async def trending():
    return {"tickers": TRENDING_TICKERS}


# ─── Prices (live via yfinance — lightweight, no torch needed) ────────────────
@app.get("/api/prices")
async def live_prices():
    """Fetch live prices for the ticker strip."""
    try:
        import yfinance as yf
    except ImportError:
        return {"prices": []}

    strip_tickers = [
        "AAPL", "NVDA", "TSLA", "MSFT", "AMZN", "GOOGL", "META",
        "NFLX", "SPY", "QQQ", "BRK-B", "JPM", "AMD", "COIN", "PLTR",
    ]

    def _fetch():
        results = []
        for sym in strip_tickers:
            try:
                t = yf.Ticker(sym)
                fi = t.fast_info
                price = getattr(fi, "last_price", None) or getattr(fi, "regularMarketPrice", None)
                prev = getattr(fi, "previous_close", None) or getattr(fi, "regularMarketPreviousClose", None)
                if price is None:
                    continue
                change = round(price - prev, 2) if prev else 0.0
                change_pct = round((change / prev) * 100, 2) if prev else 0.0
                results.append({
                    "symbol": sym,
                    "price": round(float(price), 2),
                    "change": f"{'+' if change >= 0 else ''}{change_pct}",
                    "up": change >= 0,
                })
            except Exception:
                continue
        return results

    loop = asyncio.get_event_loop()
    try:
        data = await loop.run_in_executor(executor, _fetch)
        return {"prices": data}
    except Exception:
        return {"prices": []}


# ─── Search ───────────────────────────────────────────────────────────────────
@app.get("/api/search")
async def search(
    q: str = Query(..., min_length=1, max_length=50)
):
    """Search tickers by company name using yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        # Fallback: filter trending tickers
        q_lower = q.lower()
        matches = [
            {"ticker": t["ticker"], "name": t["name"], "exchange": t["sector"], "type": "Equity"}
            for t in TRENDING_TICKERS
            if q_lower in t["ticker"].lower() or q_lower in t["name"].lower()
        ]
        return {"results": matches[:8]}

    def _search():
        q_upper = q.strip().upper()
        q_lower = q.strip().lower()

        # First check trending tickers
        matches = [
            {"ticker": t["ticker"], "name": t["name"], "exchange": t["sector"], "type": "Equity"}
            for t in TRENDING_TICKERS
            if q_lower in t["ticker"].lower() or q_lower in t["name"].lower()
        ]

        # Then try yfinance search
        try:
            results = yf.Tickers(q_upper)
            # yfinance search is limited, so trending filter is the main source
        except Exception:
            pass

        return matches[:8]

    loop = asyncio.get_event_loop()
    try:
        results = await loop.run_in_executor(executor, _search)
        return {"results": results}
    except Exception:
        return {"results": []}


# ─── Research (requires local backend) ────────────────────────────────────────
@app.post("/api/research")
async def research(req: ResearchRequest):
    """Research requires the full backend running locally."""
    raise HTTPException(
        status_code=503,
        detail="AI Research requires the local backend server. Please run: cd research_agent && uvicorn server:app --port 8001"
    )


# ─── Predict (requires local backend) ────────────────────────────────────────
@app.get("/api/predict/{ticker}")
async def predict(ticker: str):
    raise HTTPException(
        status_code=503,
        detail="AI Prediction requires the local backend server with GPU support."
    )


@app.get("/api/predict/{ticker}/history")
async def predict_history(ticker: str):
    return {"ticker": ticker.upper(), "history": []}


@app.get("/api/predict/{ticker}/backtest")
async def predict_backtest(ticker: str, days_back: int = 7):
    raise HTTPException(
        status_code=503,
        detail="Backtest requires the local backend server."
    )


# ─── Watchlist ────────────────────────────────────────────────────────────────
@app.post("/api/watchlist/summary")
async def watchlist_summary(req: WatchlistRequest):
    """Fetch live price data for a batch of tickers."""
    try:
        import yfinance as yf
    except ImportError:
        return {"stocks": []}

    def _fetch_batch():
        results = []
        for sym in req.tickers[:20]:
            try:
                t = yf.Ticker(sym)
                fi = t.fast_info
                info = t.info or {}
                price = getattr(fi, "last_price", None)
                prev = getattr(fi, "previous_close", None)
                if price is None:
                    continue
                change = round(price - prev, 2) if prev else 0.0
                change_pct = round((change / prev) * 100, 2) if prev else 0.0
                results.append({
                    "ticker": sym,
                    "name": info.get("longName") or sym,
                    "sector": info.get("sector"),
                    "price": round(float(price), 2),
                    "change": change_pct,
                    "up": change >= 0,
                })
            except Exception:
                continue
        return results

    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(executor, _fetch_batch)
    return {"stocks": data}


# ─── Compare (requires local backend) ────────────────────────────────────────
@app.post("/api/compare")
async def compare(req: CompareRequest):
    raise HTTPException(
        status_code=503,
        detail="Compare requires the full AI research backend running locally."
    )
