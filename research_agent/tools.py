"""Tools the research agent can call.

Tools:
  - get_company_news     : Alpha Vantage — recent news + per-ticker sentiment
  - get_stock_price      : yfinance — real-time price, market cap, P/E, analyst targets
  - get_technical_info   : yfinance — moving averages, volume, beta, 52-week range
  - get_earnings_info    : yfinance — EPS history, next earnings date, revenue growth
  - search_tickers       : yfinance — resolve company name → ticker symbol (no API key)
"""

import os

import requests
import yfinance as yf

BASE_URL = "https://www.alphavantage.co/query"


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1 — News & Sentiment (Alpha Vantage)
# ─────────────────────────────────────────────────────────────────────────────


def get_company_news(ticker: str, limit: int = 8) -> dict:
    """Fetch recent news articles and sentiment data for a stock ticker.

    Args:
        ticker: Stock ticker symbol, e.g. "AAPL".
        limit: Maximum number of articles to return.

    Returns:
        Dict with ticker, article count, and simplified article records including
        title, summary, source, url, overall sentiment, and per-ticker sentiment.
    """
    api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        return {"error": "ALPHA_VANTAGE_API_KEY is not set in your environment."}

    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker.upper(),
        "limit": str(limit),
        "apikey": api_key,
    }

    try:
        resp = requests.get(BASE_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        return {"error": f"Request failed: {e}"}

    if "feed" not in data:
        return {
            "error": data.get("Information")
            or data.get("Note")
            or "No news found for this ticker.",
        }

    articles = []
    for item in data["feed"][:limit]:
        ticker_match = next(
            (
                t
                for t in item.get("ticker_sentiment", [])
                if t.get("ticker") == ticker.upper()
            ),
            {},
        )
        articles.append(
            {
                "title": item.get("title"),
                "summary": item.get("summary"),
                "source": item.get("source"),
                "time_published": item.get("time_published"),
                "url": item.get("url"),
                "overall_sentiment_label": item.get("overall_sentiment_label"),
                "overall_sentiment_score": item.get("overall_sentiment_score"),
                "ticker_relevance_score": ticker_match.get("relevance_score"),
                "ticker_sentiment_label": ticker_match.get("ticker_sentiment_label"),
                "ticker_sentiment_score": ticker_match.get("ticker_sentiment_score"),
            }
        )

    return {
        "ticker": ticker.upper(),
        "article_count": len(articles),
        "articles": articles,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2 — Real-Time Price & Fundamentals (yfinance)
# ─────────────────────────────────────────────────────────────────────────────


def get_stock_price(ticker: str) -> dict:
    """Fetch real-time price and key fundamental data for a stock ticker.

    Args:
        ticker: Stock ticker symbol, e.g. "AAPL".

    Returns:
        Dict with current price, day range, 52-week range, market cap, P/E ratio,
        dividend yield, analyst price target, and recommendation.
    """
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info

        if (
            not info
            or info.get("regularMarketPrice") is None
            and info.get("currentPrice") is None
        ):
            return {
                "error": f"No price data found for ticker {ticker.upper()}. It may be invalid or delisted."
            }

        price = info.get("currentPrice") or info.get("regularMarketPrice")
        prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose")
        change_pct = (
            round(((price - prev_close) / prev_close) * 100, 2)
            if prev_close and price
            else None
        )

        return {
            "ticker": ticker.upper(),
            "company_name": info.get("longName") or info.get("shortName"),
            "exchange": info.get("exchange"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "current_price": price,
            "previous_close": prev_close,
            "change_percent_today": change_pct,
            "day_low": info.get("dayLow") or info.get("regularMarketDayLow"),
            "day_high": info.get("dayHigh") or info.get("regularMarketDayHigh"),
            "week_52_low": info.get("fiftyTwoWeekLow"),
            "week_52_high": info.get("fiftyTwoWeekHigh"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "price_to_book": info.get("priceToBook"),
            "dividend_yield": info.get("dividendYield"),
            "analyst_target_price": info.get("targetMeanPrice"),
            "analyst_recommendation": info.get("recommendationKey"),
            "analyst_count": info.get("numberOfAnalystOpinions"),
            "currency": info.get("currency", "USD"),
        }
    except Exception as e:
        return {"error": f"yfinance error for {ticker}: {str(e)}"}


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3 — Technical Indicators (yfinance)
# ─────────────────────────────────────────────────────────────────────────────


def get_technical_info(ticker: str) -> dict:
    """Fetch technical analysis data: moving averages, volume, beta, volatility.

    Args:
        ticker: Stock ticker symbol, e.g. "AAPL".

    Returns:
        Dict with 50/200-day moving averages, average volume, beta, and a simple
        trend signal based on price vs moving averages.
    """
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info

        price = info.get("currentPrice") or info.get("regularMarketPrice")
        ma50 = info.get("fiftyDayAverage")
        ma200 = info.get("twoHundredDayAverage")

        # Simple trend signal
        trend = "neutral"
        if price and ma50 and ma200:
            if price > ma50 > ma200:
                trend = "strong_uptrend"
            elif price > ma50:
                trend = "uptrend"
            elif price < ma50 < ma200:
                trend = "strong_downtrend"
            elif price < ma50:
                trend = "downtrend"

        # Price distance from moving averages (%)
        pct_from_ma50 = (
            round(((price - ma50) / ma50) * 100, 2) if price and ma50 else None
        )
        pct_from_ma200 = (
            round(((price - ma200) / ma200) * 100, 2) if price and ma200 else None
        )

        return {
            "ticker": ticker.upper(),
            "current_price": price,
            "ma_50_day": ma50,
            "ma_200_day": ma200,
            "pct_above_ma50": pct_from_ma50,
            "pct_above_ma200": pct_from_ma200,
            "trend_signal": trend,
            "avg_volume_10d": info.get("averageVolume10days")
            or info.get("averageDailyVolume10Day"),
            "avg_volume_3m": info.get("averageVolume"),
            "beta": info.get("beta"),
            "short_ratio": info.get("shortRatio"),
            "shares_short_pct": info.get("shortPercentOfFloat"),
        }
    except Exception as e:
        return {"error": f"yfinance error for {ticker}: {str(e)}"}


# ─────────────────────────────────────────────────────────────────────────────
# Tool 4 — Earnings & Financials (yfinance)
# ─────────────────────────────────────────────────────────────────────────────


def get_earnings_info(ticker: str) -> dict:
    """Fetch earnings history, upcoming earnings date, and revenue/profit growth.

    Args:
        ticker: Stock ticker symbol, e.g. "AAPL".

    Returns:
        Dict with EPS estimates vs actual (last 4 quarters), revenue growth,
        profit margins, and next earnings date.
    """
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info

        # Recent EPS history from earnings_dates
        eps_history = []
        try:
            earnings_dates = stock.earnings_dates
            if earnings_dates is not None and not earnings_dates.empty:
                for idx, row in earnings_dates.head(4).iterrows():
                    eps_history.append(
                        {
                            "date": (
                                str(idx.date()) if hasattr(idx, "date") else str(idx)
                            ),
                            "eps_estimate": (
                                row.get("EPS Estimate")
                                if "EPS Estimate" in row
                                else None
                            ),
                            "eps_actual": (
                                row.get("Reported EPS")
                                if "Reported EPS" in row
                                else None
                            ),
                            "surprise_pct": (
                                row.get("Surprise(%)") if "Surprise(%)" in row else None
                            ),
                        }
                    )
        except Exception:
            eps_history = []

        return {
            "ticker": ticker.upper(),
            "next_earnings_date": info.get("earningsDate")
            or info.get("earningsTimestamp"),
            "trailing_eps": info.get("trailingEps"),
            "forward_eps": info.get("forwardEps"),
            "revenue_growth_yoy": info.get("revenueGrowth"),
            "earnings_growth_yoy": info.get("earningsGrowth"),
            "profit_margin": info.get("profitMargins"),
            "operating_margin": info.get("operatingMargins"),
            "return_on_equity": info.get("returnOnEquity"),
            "debt_to_equity": info.get("debtToEquity"),
            "free_cashflow": info.get("freeCashflow"),
            "recent_eps_history": eps_history,
        }
    except Exception as e:
        return {"error": f"yfinance error for {ticker}: {str(e)}"}


# ─────────────────────────────────────────────────────────────────────────────
# Tool 5 — Ticker Search by Company Name (yfinance — no API key)
# ─────────────────────────────────────────────────────────────────────────────


def search_tickers(query: str, limit: int = 8) -> dict:
    """Search for stock tickers by company name or keyword.

    Use this when you have a company name but not the ticker symbol,
    e.g. query="Apple" → returns [{"ticker": "AAPL", "name": "Apple Inc.", ...}]

    Args:
        query: Company name or keyword to search for.
        limit: Max number of results to return.

    Returns:
        Dict with a list of matching tickers, their names, exchanges, and types.
    """
    last_err = None
    # Primary: yfinance Search
    try:
        search = yf.Search(query, max_results=limit)
        quotes = search.quotes or []
        results = []
        for q in quotes[:limit]:
            results.append(
                {
                    "ticker": q.get("symbol"),
                    "name": q.get("longname") or q.get("shortname"),
                    "exchange": q.get("exchange"),
                    "type": q.get("quoteType"),
                    "sector": q.get("sector"),
                }
            )
        if results:
            return {"query": query, "results": results, "count": len(results)}
    except Exception as e:
        last_err = e

    # Fallback: try treating the query as a direct ticker symbol
    try:
        q_upper = query.strip().upper()
        stock = yf.Ticker(q_upper)
        info = stock.info
        name = info.get("longName") or info.get("shortName")
        if name:
            return {
                "query": query,
                "results": [
                    {
                        "ticker": q_upper,
                        "name": name,
                        "exchange": info.get("exchange"),
                        "type": info.get("quoteType"),
                        "sector": info.get("sector"),
                    }
                ],
                "count": 1,
            }
    except Exception as e:
        last_err = last_err or e

    if last_err:
        return {"error": f"yfinance search error: {str(last_err)}"}

    return {"query": query, "results": [], "count": 0}
