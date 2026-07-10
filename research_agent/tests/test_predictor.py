"""
Tests for predictor.py v4 — covers feature engineering, sequence building,
model shapes, direct prediction, and error handling.

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
    _build_classification_targets,
    _classify_return,
    _make_sequences,
    _get_trading_dates,
    _train_lstm,
    _direct_predict,
    _interpolate_prices,
    _fetch_ohlcv,
    LSTMPredictor,
    TransformerPredictor,
    SEQ_LEN,
    PREDICT_DAYS,
    N_FEATURES,
    N_BINS,
    BIN_MIDPOINTS,
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
def trained_lstm(trending_prices):
    features = _build_feature_matrix(trending_prices)
    targets  = _build_classification_targets(trending_prices)
    X, y     = _make_sequences(features, targets)
    return _train_lstm(X, y, seed=0)


# ─── 1. Classification Helper ────────────────────────────────────────────────

class TestClassifyReturn:

    def test_strong_bear(self):
        assert _classify_return(-5.0) == 0

    def test_weak_bear(self):
        assert _classify_return(-1.0) == 1

    def test_weak_bull(self):
        assert _classify_return(1.0) == 2

    def test_strong_bull(self):
        assert _classify_return(3.0) == 3

    def test_boundary_zero(self):
        assert _classify_return(0.0) == 2  # 0% is weak bull

    def test_boundary_neg2(self):
        assert _classify_return(-2.0) == 1  # -2% is weak bear (< -2 for strong)

    def test_boundary_pos2(self):
        assert _classify_return(2.0) == 3  # +2% is strong bull


# ─── 2. Feature matrix ───────────────────────────────────────────────────────

class TestBuildFeatureMatrix:

    def test_output_shape(self, trending_prices):
        features = _build_feature_matrix(trending_prices)
        assert features.shape == (len(trending_prices), N_FEATURES)

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
        assert features.shape == (len(flat_prices), N_FEATURES)
        assert np.all(features[:, 0] == 0.0)

    def test_market_context_features_default(self, trending_prices):
        """Market context features (12-14) should have sensible defaults when no SPY/VIX data."""
        features = _build_feature_matrix(trending_prices)
        # SPY momentum (col 12) defaults to 0.5
        assert features[0, 12] == 0.5
        # VIX level (col 13) — default 20.0 / 50.0 = 0.4
        assert features[0, 13] >= 0.0
        # Relative strength (col 14) defaults to 0.5
        assert features[0, 14] == 0.5

    def test_with_market_context(self, trending_prices):
        """Test that SPY/VIX data is properly incorporated."""
        n = len(trending_prices)
        spy = np.linspace(400, 420, n)
        vix = np.full(n, 15.0)
        features = _build_feature_matrix(trending_prices, spy_prices=spy, vix_prices=vix)
        assert features.shape == (n, N_FEATURES)
        # VIX column should reflect the provided value: 15/50 = 0.3
        assert abs(features[-1, 13] - 0.3) < 1e-6


# ─── 3. Classification targets ───────────────────────────────────────────────

class TestBuildClassificationTargets:

    def test_output_keys(self, trending_prices):
        targets = _build_classification_targets(trending_prices)
        assert set(targets.keys()) == {"h1d", "h3d", "h7d"}

    def test_output_shape(self, trending_prices):
        targets = _build_classification_targets(trending_prices)
        for key in ["h1d", "h3d", "h7d"]:
            assert targets[key].shape == (len(trending_prices),)

    def test_values_in_range(self, trending_prices):
        targets = _build_classification_targets(trending_prices)
        for key in ["h1d", "h3d", "h7d"]:
            assert targets[key].min() >= 0
            assert targets[key].max() <= 3

    def test_dtype(self, trending_prices):
        targets = _build_classification_targets(trending_prices)
        for key in ["h1d", "h3d", "h7d"]:
            assert targets[key].dtype == np.int64


# ─── 4. Sequence builder ─────────────────────────────────────────────────────

class TestMakeSequences:

    def test_output_shapes(self, trending_prices):
        features = _build_feature_matrix(trending_prices)
        targets  = _build_classification_targets(trending_prices)
        X, y = _make_sequences(features, targets)
        assert X.ndim == 3
        assert X.shape[1] == SEQ_LEN
        assert X.shape[2] == N_FEATURES
        for key in ["h1d", "h3d", "h7d"]:
            assert y[key].shape[0] == X.shape[0]

    def test_tensors_are_correct_dtype(self, trending_prices):
        features = _build_feature_matrix(trending_prices)
        targets  = _build_classification_targets(trending_prices)
        X, y = _make_sequences(features, targets)
        assert X.dtype == torch.float32
        for key in ["h1d", "h3d", "h7d"]:
            assert y[key].dtype == torch.long

    def test_minimum_length_empty(self):
        short    = np.linspace(100, 110, 5)
        features = _build_feature_matrix(short)
        targets  = _build_classification_targets(short)
        X, y = _make_sequences(features, targets)
        assert X.shape[0] == 0


# ─── 5. Model architectures ──────────────────────────────────────────────────

class TestLSTMPredictor:

    def test_forward_pass_shape(self):
        model = LSTMPredictor()
        model.eval()
        for batch in [1, 4, 16]:
            x = torch.randn(batch, SEQ_LEN, N_FEATURES)
            with torch.no_grad():
                out = model(x)
            assert isinstance(out, dict)
            for key in ["h1d", "h3d", "h7d"]:
                assert out[key].shape == (batch, N_BINS)

    def test_output_is_finite(self):
        model = LSTMPredictor()
        model.eval()
        x = torch.randn(8, SEQ_LEN, N_FEATURES)
        with torch.no_grad():
            out = model(x)
        for key in ["h1d", "h3d", "h7d"]:
            assert torch.isfinite(out[key]).all()


class TestTransformerPredictor:

    def test_forward_pass_shape(self):
        model = TransformerPredictor()
        model.eval()
        for batch in [1, 4]:
            x = torch.randn(batch, SEQ_LEN, N_FEATURES)
            with torch.no_grad():
                out = model(x)
            assert isinstance(out, dict)
            for key in ["h1d", "h3d", "h7d"]:
                assert out[key].shape == (batch, N_BINS)


# ─── 6. Trading dates ────────────────────────────────────────────────────────

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


# ─── 7. Direct Prediction ────────────────────────────────────────────────────

class TestDirectPredict:

    def test_returns_all_horizons(self, trending_prices, trained_lstm):
        features = _build_feature_matrix(trending_prices)
        results = _direct_predict([trained_lstm], None, features, trending_prices)
        assert set(results.keys()) == {"h1d", "h3d", "h7d"}

    def test_result_structure(self, trending_prices, trained_lstm):
        features = _build_feature_matrix(trending_prices)
        results = _direct_predict([trained_lstm], None, features, trending_prices)
        for h in ["h1d", "h3d", "h7d"]:
            assert "probs" in results[h]
            assert "expected_return" in results[h]
            assert "predicted_price" in results[h]
            assert "direction" in results[h]
            assert len(results[h]["probs"]) == N_BINS

    def test_probabilities_sum_to_one(self, trending_prices, trained_lstm):
        features = _build_feature_matrix(trending_prices)
        results = _direct_predict([trained_lstm], None, features, trending_prices)
        for h in ["h1d", "h3d", "h7d"]:
            assert abs(sum(results[h]["probs"]) - 1.0) < 1e-5

    def test_predicted_prices_positive(self, trending_prices, trained_lstm):
        features = _build_feature_matrix(trending_prices)
        results = _direct_predict([trained_lstm], None, features, trending_prices)
        for h in ["h1d", "h3d", "h7d"]:
            assert results[h]["predicted_price"] > 0

    def test_direction_consistent(self, trending_prices, trained_lstm):
        features = _build_feature_matrix(trending_prices)
        results = _direct_predict([trained_lstm], None, features, trending_prices)
        for h in ["h1d", "h3d", "h7d"]:
            if results[h]["expected_return"] >= 0:
                assert results[h]["direction"] == "rise"
            else:
                assert results[h]["direction"] == "fall"


# ─── 8. Price Interpolation ──────────────────────────────────────────────────

class TestInterpolatePrices:

    def test_returns_7_prices(self):
        prices = _interpolate_prices(100.0, 101.0, 103.0, 107.0)
        assert len(prices) == 7

    def test_anchor_points_match(self):
        prices = _interpolate_prices(100.0, 101.0, 103.0, 107.0, hist_vol=0.0)
        assert prices[0] == 101.0  # Day 1
        assert prices[2] == 103.0  # Day 3
        assert prices[6] == 107.0  # Day 7

    def test_all_positive(self):
        prices = _interpolate_prices(100.0, 99.0, 97.0, 95.0)
        assert all(p > 0 for p in prices)

    def test_monotone_trend(self):
        """With zero jitter, a monotonically increasing anchor set should produce monotone prices."""
        prices = _interpolate_prices(100.0, 101.0, 103.0, 107.0, hist_vol=0.0)
        for i in range(1, len(prices)):
            assert prices[i] >= prices[i - 1]


# ─── 9. Error handling ───────────────────────────────────────────────────────

class TestFetchOhlcvErrors:

    def test_invalid_ticker_returns_error(self, monkeypatch):
        import yfinance as yf
        import pandas as pd

        class FakeTicker:
            def history(self, **kwargs):
                return pd.DataFrame()

        monkeypatch.setattr(yf, "Ticker", lambda t: FakeTicker())
        with pytest.raises(ValueError, match="No price data"):
            _fetch_ohlcv("FAKEXYZ")

    def test_network_failure_returns_error(self, monkeypatch):
        import yfinance as yf

        class BrokenTicker:
            def history(self, **kwargs):
                raise ConnectionError("network down")

        monkeypatch.setattr(yf, "Ticker", lambda t: BrokenTicker())
        with pytest.raises(ValueError, match="Failed to fetch data"):
            _fetch_ohlcv("AAPL")
