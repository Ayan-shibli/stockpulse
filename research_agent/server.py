"""FastAPI server — LangGraph-powered multi-tool stock research agent.

v2: WebSocket streaming, rate limiting, price caching, watchlist, comparison.
"""

import asyncio
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from agent_graph import run_research, stream_research
from tools import search_tickers
from predictor import predict_stock, get_prediction_history, backtest_stock
from sentiment_store import save_sentiment

import json

app = FastAPI(title="AstraQuant — LangGraph Research API", version="3.0.0")

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    os.environ.get("FRONTEND_URL", ""),
    "https://astraquant.vercel.app",
]
ALLOWED_ORIGINS = [o for o in ALLOWED_ORIGINS if o]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=4)

# ─── Ticker Validation Regex ─────────────────────────────────────────────────
TICKER_PATTERN = re.compile(r"^[A-Za-z0-9.\-]{1,12}$")


def validate_ticker(t: str) -> str:
    """Validate and normalise a ticker string."""
    t = t.strip().upper()
    if not t:
        raise ValueError("Ticker is required.")
    if not TICKER_PATTERN.match(t):
        raise ValueError(f"Invalid ticker format: '{t}'. Use 1-12 alphanumeric characters.")
    return t


# ─── Rate Limiter Middleware ─────────────────────────────────────────────────
# Sliding-window in-memory rate limiter, keyed by client IP.
RATE_LIMITS = {
    # Heavy endpoints
    "/api/research":  {"max_requests": 30, "window_seconds": 60},
    "/api/compare":   {"max_requests": 20, "window_seconds": 60},
    # Medium endpoints
    "/api/predict":   {"max_requests": 30, "window_seconds": 60},
    # Light endpoints — default
    "_default":       {"max_requests": 60, "window_seconds": 60},
}

# Store: {ip: {path_key: [timestamps]}}
_rate_store: dict = defaultdict(lambda: defaultdict(list))


def _get_rate_key(path: str) -> str:
    """Map a request path to its rate-limit bucket."""
    if path.startswith("/api/research"):
        return "/api/research"
    if path.startswith("/api/compare"):
        return "/api/compare"
    if path.startswith("/api/predict"):
        return "/api/predict"
    return "_default"


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Skip non-API and WebSocket requests
        path = request.url.path
        if not path.startswith("/api") or request.scope.get("type") == "websocket":
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        rate_key  = _get_rate_key(path)
        limits    = RATE_LIMITS.get(rate_key, RATE_LIMITS["_default"])
        max_req   = limits["max_requests"]
        window    = limits["window_seconds"]
        now       = time.time()

        # Clean old entries
        timestamps = _rate_store[client_ip][rate_key]
        timestamps[:] = [ts for ts in timestamps if now - ts < window]

        if len(timestamps) >= max_req:
            retry_after = int(window - (now - timestamps[0])) + 1
            return JSONResponse(
                status_code=429,
                content={"detail": f"Rate limit exceeded. Try again in {retry_after}s."},
                headers={"Retry-After": str(retry_after)},
            )

        timestamps.append(now)
        return await call_next(request)


app.add_middleware(RateLimitMiddleware)


# ─── Price Cache (background refresh) ────────────────────────────────────────
# Pre-fetch prices on startup so the ticker strip loads instantly.
_price_cache: dict = {"prices": [], "updated_at": 0}
_price_lock = asyncio.Lock()

# Curated trending list — used as fallback / initial suggestions
TRENDING_TICKERS = [
    # ── Top USA Tech & AI ────────────────────────────────────────────────────
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
    # ── USA Finance ──────────────────────────────────────────────────────────
    {"ticker": "JPM", "name": "JPMorgan Chase", "sector": "Finance"},
    {"ticker": "GS", "name": "Goldman Sachs Group", "sector": "Finance"},
    {"ticker": "BAC", "name": "Bank of America", "sector": "Finance"},
    {"ticker": "V", "name": "Visa Inc.", "sector": "Fintech"},
    {"ticker": "MA", "name": "Mastercard Inc.", "sector": "Fintech"},
    {"ticker": "PYPL", "name": "PayPal Holdings", "sector": "Fintech"},
    {"ticker": "SQ", "name": "Block Inc.", "sector": "Fintech"},
    {"ticker": "COIN", "name": "Coinbase Global", "sector": "Crypto"},
    # ── USA Growth & Consumer ────────────────────────────────────────────────
    {"ticker": "NFLX", "name": "Netflix Inc.", "sector": "Streaming"},
    {"ticker": "DIS", "name": "The Walt Disney Co.", "sector": "Entertainment"},
    {"ticker": "SPOT", "name": "Spotify Technology", "sector": "Music / Media"},
    {"ticker": "UBER", "name": "Uber Technologies", "sector": "Mobility"},
    {"ticker": "LYFT", "name": "Lyft Inc.", "sector": "Mobility"},
    {"ticker": "ABNB", "name": "Airbnb Inc.", "sector": "Travel / Hospitality"},
    {"ticker": "SHOP", "name": "Shopify Inc.", "sector": "E-Commerce"},
    # ── USA Cloud & Cybersecurity ────────────────────────────────────────────
    {"ticker": "SNOW", "name": "Snowflake Inc.", "sector": "Cloud / Data"},
    {"ticker": "CRWD", "name": "CrowdStrike Holdings", "sector": "Cybersecurity"},
    {"ticker": "NET", "name": "Cloudflare Inc.", "sector": "Cybersecurity"},
    {"ticker": "DDOG", "name": "Datadog Inc.", "sector": "Cloud / Monitoring"},
    {"ticker": "ZM", "name": "Zoom Video Communications", "sector": "SaaS / Comms"},
    {"ticker": "ORCL", "name": "Oracle Corporation", "sector": "Cloud / Enterprise"},
    {"ticker": "ADBE", "name": "Adobe Inc.", "sector": "Software"},
    {"ticker": "CRM", "name": "Salesforce Inc.", "sector": "CRM / SaaS"},
    # ── USA Industrials & Energy ─────────────────────────────────────────────
    {"ticker": "BA", "name": "Boeing Co.", "sector": "Aerospace"},
    {"ticker": "RIVN", "name": "Rivian Automotive", "sector": "EV"},
    {"ticker": "XOM", "name": "ExxonMobil Corp.", "sector": "Energy / Oil"},
    # ── Pakistan (KSE) ───────────────────────────────────────────────────────
    {"ticker": "HBL.KA", "name": "Habib Bank Limited", "sector": "Pakistan / Banking"},
    {"ticker": "LUCK.KA", "name": "Lucky Cement Ltd.", "sector": "Pakistan / Cement"},
    {
        "ticker": "ENGRO.KA",
        "name": "Engro Corporation",
        "sector": "Pakistan / Conglomerate",
    },
    {"ticker": "PSO.KA", "name": "Pakistan State Oil", "sector": "Pakistan / Energy"},
    {"ticker": "OGDC.KA", "name": "Oil & Gas Dev. Co.", "sector": "Pakistan / Energy"},
    {
        "ticker": "PPL.KA",
        "name": "Pakistan Petroleum Ltd.",
        "sector": "Pakistan / Energy",
    },
    {"ticker": "MARI.KA", "name": "Mari Petroleum Co.", "sector": "Pakistan / Energy"},
    {"ticker": "MCB.KA", "name": "MCB Bank Limited", "sector": "Pakistan / Banking"},
    {"ticker": "UBL.KA", "name": "United Bank Limited", "sector": "Pakistan / Banking"},
    {"ticker": "TRG.KA", "name": "TRG Pakistan Ltd.", "sector": "Pakistan / IT"},
    {"ticker": "SYS.KA", "name": "Systems Limited", "sector": "Pakistan / IT"},
    {
        "ticker": "EFERT.KA",
        "name": "Engro Fertilizers",
        "sector": "Pakistan / Fertilizers",
    },
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
        if len(v) > 20:
            raise ValueError("Ticker/query too long (max 20 chars).")
        return v


class CompareRequest(BaseModel):
    ticker_a: str
    ticker_b: str

    @field_validator("ticker_a", "ticker_b")
    @classmethod
    def validate_tickers(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Ticker is required.")
        if len(v) > 20:
            raise ValueError("Ticker/query too long.")
        return v


class WatchlistRequest(BaseModel):
    tickers: List[str]

    @field_validator("tickers")
    @classmethod
    def validate_list(cls, v):
        if len(v) > 20:
            raise ValueError("Maximum 20 tickers per watchlist request.")
        return [t.strip().upper() for t in v if t.strip()]


# ─── Ticker Strip Prices (live, with caching) ────────────────────────────────
STRIP_TICKERS = [
    "AAPL", "NVDA", "TSLA", "MSFT", "AMZN", "GOOGL", "META",
    "NFLX", "SPY", "QQQ", "BRK-B", "JPM", "AMD", "COIN", "PLTR",
]


def _fetch_strip_prices():
    """Blocking function — fetch live prices for ticker strip."""
    import yfinance as yf
    results = []
    for sym in STRIP_TICKERS:
        try:
            t  = yf.Ticker(sym)
            fi = t.fast_info
            price = getattr(fi, "last_price", None) or getattr(fi, "regularMarketPrice", None)
            prev  = getattr(fi, "previous_close", None) or getattr(fi, "regularMarketPreviousClose", None)
            if price is None:
                continue
            change = round(price - prev, 2) if prev else 0.0
            change_pct = round((change / prev) * 100, 2) if prev else 0.0
            results.append({
                "symbol": sym,
                "price":  round(float(price), 2),
                "change": f"{'+' if change >= 0 else ''}{change_pct}",
                "up":     change >= 0,
            })
        except Exception:
            continue
    return results


async def _refresh_price_cache():
    """Background task that refreshes the price cache every 60s."""
    global _price_cache
    while True:
        try:
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(executor, _fetch_strip_prices)
            if data:
                async with _price_lock:
                    _price_cache = {"prices": data, "updated_at": time.time()}
        except Exception:
            pass
        await asyncio.sleep(60)


@app.on_event("startup")
async def startup_event():
    """Pre-fetch prices and start background refresh."""
    asyncio.create_task(_refresh_price_cache())


@app.get("/api/prices")
async def live_prices():
    """Return cached prices. Instant response since cache is pre-warmed."""
    async with _price_lock:
        cached = _price_cache.copy()
    if cached["prices"]:
        return {"prices": cached["prices"]}
    # If cache is empty (very first call before background task completes),
    # do a synchronous fetch as fallback.
    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(executor, _fetch_strip_prices)
    return {"prices": data}


# ─── Health ───────────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "AstraQuant LangGraph Agent", "version": "3.0.0"}


# ─── Trending ─────────────────────────────────────────────────────────────────
@app.get("/api/trending")
async def trending():
    return {"tickers": TRENDING_TICKERS}


# ─── Ticker Search (yfinance — no extra API key) ──────────────────────────────
@app.get("/api/search")
async def search(
    q: str = Query(..., min_length=1, max_length=50, description="Company name or ticker")
):
    """Search for tickers by company name. Covers 10,000+ symbols via yfinance."""
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            executor,
            lambda: search_tickers(q.strip(), limit=8),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    return result


# ─── Research (LangGraph agent — HTTP) ────────────────────────────────────────
@app.post("/api/research")
async def research(req: ResearchRequest):
    """Run the LangGraph multi-tool research agent on a ticker or company name."""
    query = req.ticker.strip()
    if not query:
        raise HTTPException(
            status_code=400, detail="Ticker or company name is required."
        )

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            executor,
            lambda: run_research(query, verbose=False),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if "error" in result and len(result) <= 2:
        raise HTTPException(status_code=422, detail=result["error"])

    # ── Save sentiment to cache so LSTM can use it as feature ────────────────
    try:
        score = result.get("sentiment_score")
        label = result.get("sentiment", "neutral")
        ticker_resolved = result.get("ticker") or query
        if score is not None:
            save_sentiment(ticker_resolved, score, label)
    except Exception:
        pass  # never block the response due to sentiment caching failure

    return result


# ─── Research WebSocket (live streaming) ──────────────────────────────────────
@app.websocket("/ws/research")
async def ws_research(websocket: WebSocket):
    """WebSocket endpoint that streams research steps live to the client."""
    await websocket.accept()
    try:
        # Wait for the client to send a ticker
        raw = await websocket.receive_text()
        try:
            msg = json.loads(raw)
            ticker = msg.get("ticker", "").strip()
        except (json.JSONDecodeError, AttributeError):
            ticker = raw.strip()

        if not ticker:
            await websocket.send_json({"type": "error", "data": "Ticker is required."})
            await websocket.close()
            return

        # Stream research steps from the agent graph
        loop = asyncio.get_event_loop()

        def _run_stream():
            return list(stream_research(ticker))

        # We collect from the generator in the executor, yielding steps
        # We use a queue to bridge the sync generator → async websocket
        import queue
        step_queue: queue.Queue = queue.Queue()
        final_result = [None]

        def _run_and_queue():
            try:
                for item in stream_research(ticker):
                    step_queue.put(item)
                step_queue.put(None)  # sentinel
            except Exception as e:
                step_queue.put({"type": "error", "data": str(e)})
                step_queue.put(None)

        # Run in thread
        import threading
        thread = threading.Thread(target=_run_and_queue, daemon=True)
        thread.start()

        # Read from queue and send to WebSocket
        while True:
            try:
                item = await loop.run_in_executor(None, lambda: step_queue.get(timeout=30))
            except Exception:
                break
            if item is None:
                break
            await websocket.send_json(item)

        # Save sentiment if available
        try:
            if final_result[0]:
                score = final_result[0].get("sentiment_score")
                label = final_result[0].get("sentiment", "neutral")
                tk    = final_result[0].get("ticker") or ticker
                if score is not None:
                    save_sentiment(tk, score, label)
        except Exception:
            pass

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "data": str(e)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ─── Predict (PyTorch LSTM) ───────────────────────────────────────────────────
@app.get("/api/predict/{ticker}")
async def predict(ticker: str):
    """Run a PyTorch LSTM to forecast the next 7 trading days for a ticker."""
    try:
        t = validate_ticker(ticker)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            executor,
            lambda: predict_stock(t),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    return result


# ─── Prediction History (past predictions + outcomes) ─────────────────────────
@app.get("/api/predict/{ticker}/history")
async def predict_history(ticker: str):
    """Return stored predictions + actual-price outcomes for a ticker."""
    try:
        t = validate_ticker(ticker)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    loop = asyncio.get_event_loop()
    try:
        history = await loop.run_in_executor(
            executor,
            lambda: get_prediction_history(t),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"ticker": t, "history": history}


# ─── Backtest (instant accuracy report) ──────────────────────────────────────
@app.get("/api/predict/{ticker}/backtest")
async def predict_backtest(
    ticker: str,
    days_back: int = Query(
        7,
        ge=3,
        le=30,
        description="How many recent trading days to withhold for testing",
    ),
):
    """
    Simulate a prediction made `days_back` trading days ago.
    Trains LSTM on data UP TO that point, predicts forward, then
    compares to actual prices we already have → instant accuracy report.
    """
    try:
        t = validate_ticker(ticker)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            executor,
            lambda: backtest_stock(t, days_back=days_back),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    return result


# ─── Watchlist Summary (batch price fetch) ────────────────────────────────────
@app.post("/api/watchlist/summary")
async def watchlist_summary(req: WatchlistRequest):
    """Fetch live price data for a batch of tickers (used by the watchlist dashboard)."""
    import yfinance as yf

    def _fetch_batch():
        results = []
        for sym in req.tickers[:20]:
            try:
                t  = yf.Ticker(sym)
                fi = t.fast_info
                info = t.info or {}
                price = getattr(fi, "last_price", None) or getattr(fi, "regularMarketPrice", None)
                prev  = getattr(fi, "previous_close", None) or getattr(fi, "regularMarketPreviousClose", None)
                if price is None:
                    continue
                change     = round(price - prev, 2) if prev else 0.0
                change_pct = round((change / prev) * 100, 2) if prev else 0.0
                results.append({
                    "ticker":      sym,
                    "name":        info.get("longName") or info.get("shortName") or sym,
                    "sector":      info.get("sector"),
                    "price":       round(float(price), 2),
                    "change":      change_pct,
                    "up":          change >= 0,
                    "market_cap":  info.get("marketCap"),
                    "pe_ratio":    info.get("trailingPE"),
                })
            except Exception:
                continue
        return results

    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(executor, _fetch_batch)
    return {"stocks": data}


# ─── Compare (side-by-side research) ──────────────────────────────────────────
@app.post("/api/compare")
async def compare(req: CompareRequest):
    """Run research on two tickers concurrently and return both results."""
    a = req.ticker_a.strip()
    b = req.ticker_b.strip()
    if not a or not b:
        raise HTTPException(status_code=400, detail="Both tickers are required.")

    loop = asyncio.get_event_loop()
    try:
        result_a, result_b = await asyncio.gather(
            loop.run_in_executor(executor, lambda: run_research(a, verbose=False)),
            loop.run_in_executor(executor, lambda: run_research(b, verbose=False)),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Save sentiment for both
    for res, q in [(result_a, a), (result_b, b)]:
        try:
            score = res.get("sentiment_score")
            label = res.get("sentiment", "neutral")
            tk    = res.get("ticker") or q
            if score is not None:
                save_sentiment(tk, score, label)
        except Exception:
            pass

    return {"ticker_a": result_a, "ticker_b": result_b}
