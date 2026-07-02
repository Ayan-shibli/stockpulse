"""Tests for sentiment_store.py."""

import os
import pytest
import sys
from datetime import datetime, timedelta

# Ensure backend folder is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import sentiment_store


@pytest.fixture
def mock_cache_path(tmp_path):
    """Fixture to temporarily replace the cache path with a temp file."""
    temp_file = tmp_path / "temp_sentiment_cache.json"
    original_path = sentiment_store.SENTIMENT_CACHE_PATH
    sentiment_store.SENTIMENT_CACHE_PATH = str(temp_file)
    yield temp_file
    sentiment_store.SENTIMENT_CACHE_PATH = original_path


def test_empty_cache(mock_cache_path):
    """Ensure an unknown ticker returns None and doesn't crash."""
    score = sentiment_store.get_sentiment("NONEXISTENT")
    assert score is None
    assert sentiment_store.get_all_sentiments() == {}


def test_save_and_retrieve_sentiment(mock_cache_path):
    """Ensure that we can save a sentiment entry and retrieve it correctly."""
    ticker = "AAPL"
    score = 0.85
    label = "bullish"

    # Save
    sentiment_store.save_sentiment(ticker=ticker, score=score, label=label)

    # Retrieve
    retrieved_score = sentiment_store.get_sentiment(ticker)
    assert retrieved_score == 0.85

    # Retrieve all
    all_sents = sentiment_store.get_all_sentiments()
    assert ticker in all_sents
    assert all_sents[ticker]["score"] == 0.85
    assert all_sents[ticker]["label"] == "bullish"
    assert "updated_at" in all_sents[ticker]


def test_sentiment_stale_ttl(mock_cache_path):
    """Ensure that get_sentiment returns None if the entry is older than TTL."""
    ticker = "TSLA"
    
    # Save a fresh entry
    sentiment_store.save_sentiment(ticker, 0.45, "bullish")
    assert sentiment_store.get_sentiment(ticker) == 0.45

    # Mock an old timestamp (25 hours ago, since TTL is 24 hours)
    cache = sentiment_store._load_cache()
    old_time = (datetime.utcnow() - timedelta(hours=25)).strftime("%Y-%m-%d %H:%M UTC")
    cache[ticker]["updated_at"] = old_time
    sentiment_store._save_cache(cache)

    # Should be stale now
    assert sentiment_store.get_sentiment(ticker) is None
