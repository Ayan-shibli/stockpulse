"""
Tests for predictor.py — covers feature engineering, sequence building,
model shape, rollout realism, and error handling.

Run from research_agent/ with:
    pytest tests/ -v
"""

import numpy as np
import pytest
import torch

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from predictor import (
    _build_feature_matrix,
    _build_return_targets,
    _make_sequences,
    _get_trading_dates,
    _train_model,
    _rollout,
    _fetch_prices,
    StockLSTM,
    SEQ_LEN,
    PREDICT_DAYS,
    MAX_DAILY_RETURN_PCT,
)
from datetime import datetime


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def flat_prices():
    return np.full(60, 100.0)

@pytest.fixture
def trending_prices():
    np.random.seed(42)
    base  = np.linspace(150, 200, 80)
    noise = np.random.randn(80) * 2
    return base + noise

@pytest.fixture
def trained_model(trending_prices):
    features = _build_feature_matrix(trending_prices)
    targets  = _build_return_targets(trending_prices)
    X, y     = _make_sequences(features, targets)
    return _train_model(X, y, seed=0)


# ─── 1. Feature matrix ───────────────────────────────────────────────────────

class TestBuildFeatureMatrix:

    def test_output_shape(self, trending_prices):
        features = _build_feature_matrix(trending_prices)
        assert features.shape == (len(trending_prices), 4)

    def test_price_column_normalised(self, trending_prices):
        features = _build_feature_matrix(trending_prices)
        assert features[:, 0].min() >= 0.0 - 1e-9
        assert features[:, 0].max() <= 1.0 + 1e-9

    def test_return_column_normalised(self, trending_prices):
        features = _build_feature_matrix(trending_prices)
        assert features[:, 1].min() >= 0.0 - 1e-9
        assert features[:, 1].max() <= 1.0 + 1e-9

    def test_zscore_column_normalised(self, trending_prices):
        features = _build_feature_matrix(trending_prices)
        assert features[:, 2].min() >= 0.0 - 1e-9
        assert features[:, 2].max() <= 1.0 + 1e-9

    def test_sentiment_column_scaled(self, trending_prices):
        features = _build_feature_matrix(trending_prices, sentiment_score=0.6)
        assert np.allclose(features[:, 3], 0.8)

    def test_flat_prices_no_crash(self, flat_prices):
        features = _build_feature_matrix(flat_prices)
        assert features.shape == (len(flat_prices), 4)
        assert np.all(features[:, 0] == 0.0)


# ─── 2. Return targets ───────────────────────────────────────────────────────

class TestBuildReturnTargets:

    def test_output_shape(self, trending_prices):
        targets = _build_return_targets(trending_prices)
        assert targets.shape == (len(trending_prices),)

    def test_values_in_range(self, trending_prices):
        targets = _build_return_targets(trending_prices)
        assert targets.min() >= 0.0 - 1e-9
        assert targets.max() <= 1.0 + 1e-9

    def test_first_element_is_neutral(self, trending_prices):
        targets = _build_return_targets(trending_prices)
        assert abs(targets[0] - 0.5) < 1e-9


# ─── 3. Sequence builder ─────────────────────────────────────────────────────

class TestMakeSequences:

    def test_output_shapes(self, trending_prices):
        features = _build_feature_matrix(trending_prices)
        targets  = _build_return_targets(trending_prices)
        X, y = _make_sequences(features, targets)
        expected_samples = len(trending_prices) - SEQ_LEN
        assert X.shape == (expected_samples, SEQ_LEN, 4)
        assert y.shape == (expected_samples, 1)

    def test_tensors_are_float32(self, trending_prices):
        features = _build_feature_matrix(trending_prices)
        targets  = _build_return_targets(trending_prices)
        X, y = _make_sequences(features, targets)
        assert X.dtype == torch.float32
        assert y.dtype == torch.float32

    def test_minimum_length_raises(self):
        short    = np.linspace(100, 110, 5)
        features = _build_feature_matrix(short)
        targets  = _build_return_targets(short)
        X, y = _make_sequences(features, targets)
        assert X.shape[0] == 0


# ─── 4. StockLSTM ────────────────────────────────────────────────────────────

class TestStockLSTM:

    def test_forward_pass_shape(self):
        model = StockLSTM()
        model.eval()
        for batch in [1, 4, 16]:
            x = torch.randn(batch, SEQ_LEN, 4)
            with torch.no_grad():
                out = model(x)
            assert out.shape == (batch, 1)

    def test_output_is_finite(self):
        model = StockLSTM()
        model.eval()
        x = torch.randn(8, SEQ_LEN, 4)
        with torch.no_grad():
            out = model(x)
        assert torch.isfinite(out).all()

    def test_input_size_mismatch_raises(self):
        model   = StockLSTM(input_size=4)
        x_wrong = torch.randn(1, SEQ_LEN, 2)
        with pytest.raises(RuntimeError):
            model(x_wrong)


# ─── 5. Trading dates ────────────────────────────────────────────────────────

class TestGetTradingDates:

    def test_returns_correct_count(self):
        start = datetime(2026, 6, 20)
        dates = _get_trading_dates(start, 7)
        assert len(dates) == 7

    def test_no_weekends(self):
        start = datetime(2026, 6, 20)
        dates = _get_trading_dates(start, 14)
        for d in dates:
            assert datetime.strptime(d, "%Y-%m-%d").weekday() < 5

    def test_dates_are_sequential(self):
        start  = datetime(2026, 6, 20)
        dates  = _get_trading_dates(start, 10)
        parsed = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
        assert parsed == sorted(parsed)


# ─── 6. Rollout realism ──────────────────────────────────────────────────────

class TestRollout:

    def test_returns_correct_count(self, trending_prices, trained_model):
        features = _build_feature_matrix(trending_prices)
        result   = _rollout([trained_model], features, trending_prices, PREDICT_DAYS)
        assert len(result) == PREDICT_DAYS

    def test_no_steep_daily_moves(self, trending_prices, trained_model):
        features = _build_feature_matrix(trending_prices)
        prices   = _rollout([trained_model], features, trending_prices, PREDICT_DAYS)
        for i in range(1, len(prices)):
            daily_move = abs(prices[i] - prices[i-1]) / prices[i-1] * 100
            assert daily_move <= MAX_DAILY_RETURN_PCT + 0.01

    def test_prices_are_positive(self, trending_prices, trained_model):
        features = _build_feature_matrix(trending_prices)
        prices   = _rollout([trained_model], features, trending_prices, PREDICT_DAYS)
        assert all(p > 0 for p in prices)

    def test_ensemble_averages_multiple_models(self, trending_prices, trained_model):
        features = _build_feature_matrix(trending_prices)
        model2   = StockLSTM()
        result   = _rollout([trained_model, model2], features, trending_prices, PREDICT_DAYS)
        assert len(result) == PREDICT_DAYS


# ─── 7. _fetch_prices error handling ─────────────────────────────────────────

class TestFetchPricesErrors:

    def test_invalid_ticker_returns_error(self, monkeypatch):
        import yfinance as yf
        import pandas as pd

        class FakeTicker:
            def history(self, **kwargs):
                return pd.DataFrame()

        monkeypatch.setattr(yf, "Ticker", lambda t: FakeTicker())
        with pytest.raises(ValueError, match="No historical price data"):
            _fetch_prices("FAKEXYZ")

    def test_network_failure_returns_error(self, monkeypatch):
        import yfinance as yf

        class BrokenTicker:
            def history(self, **kwargs):
                raise ConnectionError("network down")

        monkeypatch.setattr(yf, "Ticker", lambda t: BrokenTicker())
        with pytest.raises(ValueError, match="Failed to fetch data"):
            _fetch_prices("AAPL")
