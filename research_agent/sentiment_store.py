"""
Sentiment Store — bridges the LangGraph agent and the LSTM predictor.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional

SENTIMENT_CACHE_PATH = os.path.join(
    os.path.dirname(__file__), "sentiment_cache.json"
)

SENTIMENT_TTL_HOURS = 24


def _load_cache() -> dict:
    if os.path.exists(SENTIMENT_CACHE_PATH):
        try:
            with open(SENTIMENT_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_cache(cache: dict) -> None:
    try:
        with open(SENTIMENT_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass


def save_sentiment(ticker: str, score: float, label: str) -> None:
    cache = _load_cache()
    cache[ticker.upper()] = {
        "score":      round(float(score), 4),
        "label":      label,
        "updated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    }
    _save_cache(cache)


def get_sentiment(ticker: str) -> Optional[float]:
    cache = _load_cache()
    entry = cache.get(ticker.upper())
    if not entry:
        return None

    try:
        updated = datetime.strptime(entry["updated_at"], "%Y-%m-%d %H:%M UTC")
        age_h   = (datetime.utcnow() - updated).total_seconds() / 3600
        if age_h > SENTIMENT_TTL_HOURS:
            return None
    except Exception:
        return None

    score = entry.get("score")
    if score is None:
        return None

    return float(max(-1.0, min(1.0, score)))


def get_all_sentiments() -> dict:
    return _load_cache()
