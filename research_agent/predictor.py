"""
AstraQuant Predictor v4 — Multi-Horizon Classification Ensemble

Architecture:
  - LSTMPredictor: 2-layer LSTM + 3 classification heads (1d/3d/7d)
  - TransformerPredictor: multi-head self-attention + 3 classification heads
  - XGBoost: gradient boosted trees (3 classifiers, one per horizon)
  - Ensemble: weighted average of all models' probabilities

Key improvements over v3:
  1. No autoregressive rollout — predicts Day 1, 3, 7 directly from real data
  2. Market context features (SPY momentum, VIX fear, relative strength)
  3. Classification (4 return bins) instead of regression
  4. XGBoost ensemble member for diversity

Features (15 total):
  0  normalised price
  1  daily return
  2  5-day z-score
  3  NLP sentiment (from LangGraph agent)
  4  realised volatility (20-day)
  5  RSI-14
  6  MACD histogram
  7  Bollinger Band position
  8  SMA-20 distance
  9  rate of change (10-day)
  10 volume ratio
  11 macro fear index
  12 SPY 5-day momentum (market trend)
  13 VIX level (fear index)
  14 Relative strength vs SPY

Classification bins (per horizon):
  Bin 0: Strong Bear  (return < -2%)
  Bin 1: Weak Bear    (return -2% to 0%)
  Bin 2: Weak Bull    (return 0% to +2%)
  Bin 3: Strong Bull  (return > +2%)
"""

from __future__ import annotations
import logging

logging.getLogger("yfinance").setLevel(logging.CRITICAL)

import json
import math
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import yfinance as yf

from sentiment_store import get_sentiment

# ─── XGBoost (optional — graceful fallback) ───────────────────────────────────
try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


# ─── Hyperparameters ─────────────────────────────────────────────────────────
SEQ_LEN = 20
N_FEATURES = 15  # 12 original + 3 market context
PREDICT_DAYS = 7
CACHE_HOURS = 24
MAX_DAILY_RETURN_PCT = 5.0
HISTORY_PERIOD = "2y"

# Classification bins
N_BINS = 4
BIN_EDGES = [-float("inf"), -2.0, 0.0, 2.0, float("inf")]
BIN_MIDPOINTS = np.array([-3.0, -1.0, 1.0, 3.0])  # representative return per bin
HORIZONS = {"h1d": 1, "h3d": 3, "h7d": 7}

# LSTM settings
LSTM_HIDDEN = 64
LSTM_LAYERS = 2
LSTM_EPOCHS = 20   # reduced from 50 — fast enough for first-load, cache handles repeats
LSTM_LR = 0.005

# Transformer settings
TF_DIM = 64  # model dimension
TF_HEADS = 4  # attention heads
TF_LAYERS = 2  # encoder layers
TF_EPOCHS = 15   # reduced from 40
TF_LR = 0.003

# Ensemble: 2 LSTMs + 1 Transformer + (optional) XGBoost
N_LSTM = 2
N_TRANSFORM = 1

# Neural net vs XGBoost blend weights
NN_WEIGHT = 0.6
XGB_WEIGHT = 0.4

LOG_PATH = os.path.join(os.path.dirname(__file__), "predictions_log.json")
MODEL_CACHE_DIR = os.path.join(os.path.dirname(__file__), "model_cache")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)


# ─── Classification Helper ───────────────────────────────────────────────────
def _classify_return(pct_return: float) -> int:
    """Assign a percentage return to one of 4 bins."""
    if pct_return < -2.0:
        return 0  # strong bear
    elif pct_return < 0.0:
        return 1  # weak bear
    elif pct_return < 2.0:
        return 2  # weak bull
    else:
        return 3  # strong bull


# ─── Model 1: LSTM ───────────────────────────────────────────────────────────
class LSTMPredictor(nn.Module):
    """
    2-layer LSTM with 3 classification heads for multi-horizon prediction.
    Each head outputs 4-class logits (one per return bin).
    """

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            N_FEATURES,
            LSTM_HIDDEN,
            LSTM_LAYERS,
            batch_first=True,
            dropout=0.2 if LSTM_LAYERS > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(LSTM_HIDDEN)
        self.shared = nn.Sequential(
            nn.Linear(LSTM_HIDDEN, LSTM_HIDDEN // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        # 3 classification heads — one per forecast horizon
        self.head_1d = nn.Linear(LSTM_HIDDEN // 2, N_BINS)
        self.head_3d = nn.Linear(LSTM_HIDDEN // 2, N_BINS)
        self.head_7d = nn.Linear(LSTM_HIDDEN // 2, N_BINS)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out, _ = self.lstm(x)
        last = self.norm(out[:, -1, :])
        h = self.shared(last)
        return {
            "h1d": self.head_1d(h),
            "h3d": self.head_3d(h),
            "h7d": self.head_7d(h),
        }


# ─── Model 2: Transformer ────────────────────────────────────────────────────
class TransformerPredictor(nn.Module):
    """
    Multi-head self-attention encoder with 3 classification heads.
    Attends to ALL positions simultaneously — catches long-range patterns
    that LSTM might miss (e.g., RSI divergence 15 days ago).
    """

    def __init__(self):
        super().__init__()
        # Project input features to model dimension
        self.input_proj = nn.Linear(N_FEATURES, TF_DIM)

        # Positional encoding — tells the model where each day is in the window
        pe = torch.zeros(SEQ_LEN, TF_DIM)
        pos = torch.arange(SEQ_LEN).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, TF_DIM, 2).float() * (-math.log(10000.0) / TF_DIM)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, SEQ_LEN, TF_DIM)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=TF_DIM,
            nhead=TF_HEADS,
            dim_feedforward=TF_DIM * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=TF_LAYERS)

        # Shared trunk
        self.norm = nn.LayerNorm(TF_DIM)
        self.shared = nn.Sequential(
            nn.Linear(TF_DIM, TF_DIM // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # 3 classification heads
        self.head_1d = nn.Linear(TF_DIM // 2, N_BINS)
        self.head_3d = nn.Linear(TF_DIM // 2, N_BINS)
        self.head_7d = nn.Linear(TF_DIM // 2, N_BINS)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: (batch, SEQ_LEN, N_FEATURES)
        h = self.input_proj(x) + self.pe  # add positional encoding
        h = self.encoder(h)
        out = self.norm(h[:, -1, :])  # last timestep
        out = self.shared(out)
        return {
            "h1d": self.head_1d(out),
            "h3d": self.head_3d(out),
            "h7d": self.head_7d(out),
        }


# ─── Technical Indicators ────────────────────────────────────────────────────
def _rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(prices)
    out = np.full(n, 0.5)
    if n < period + 1:
        return out
    deltas = np.diff(prices)
    for i in range(period, n - 1):
        w = deltas[max(0, i - period) : i]
        gains = w[w > 0].mean() if (w > 0).any() else 0.0
        loss = (-w[w < 0]).mean() if (w < 0).any() else 1e-10
        rs = gains / loss
        out[i + 1] = rs / (1 + rs)
    return out


def _macd(prices: np.ndarray) -> np.ndarray:
    n = len(prices)
    out = np.full(n, 0.5)
    if n < 26:
        return out

    def ema(data, span):
        a, r = 2 / (span + 1), np.zeros(len(data))
        r[0] = data[0]
        for i in range(1, len(data)):
            r[i] = a * data[i] + (1 - a) * r[i - 1]
        return r

    hist = ema(prices, 12) - ema(prices, 26)
    sig = ema(hist, 9)
    h = hist - sig
    mx = np.abs(h).max()
    if mx > 0:
        out = np.clip((h / mx + 1) / 2, 0, 1)
    return out


def _bollinger(prices: np.ndarray, period: int = 20) -> np.ndarray:
    n = len(prices)
    out = np.full(n, 0.5)
    for i in range(period - 1, n):
        w = prices[i - period + 1 : i + 1]
        mu, s = w.mean(), w.std()
        if s > 0:
            out[i] = np.clip((prices[i] - (mu - 2 * s)) / (4 * s), 0, 1)
    return out


def _sma_dist(prices: np.ndarray, period: int = 20) -> np.ndarray:
    n = len(prices)
    out = np.full(n, 0.5)
    for i in range(period - 1, n):
        sma = prices[i - period + 1 : i + 1].mean()
        if sma > 0:
            out[i] = np.clip((prices[i] - sma) / sma * 100 / 10 + 0.5, 0, 1)
    return out


def _roc(prices: np.ndarray, period: int = 10) -> np.ndarray:
    n = len(prices)
    out = np.full(n, 0.5)
    for i in range(period, n):
        if prices[i - period] > 0:
            pct = (prices[i] - prices[i - period]) / prices[i - period] * 100
            out[i] = np.clip(pct / 20 + 0.5, 0, 1)
    return out


def _vol_ratio(volumes: np.ndarray, period: int = 20) -> np.ndarray:
    n = len(volumes)
    out = np.full(n, 0.5)
    for i in range(period, n):
        avg = volumes[i - period : i].mean()
        if avg > 0:
            out[i] = np.clip(volumes[i] / avg / 3.0, 0, 1)
    return out


# ─── Feature Matrix ───────────────────────────────────────────────────────────
def _build_feature_matrix(
    prices: np.ndarray,
    sentiment_score: float = 0.0,
    volumes: Optional[np.ndarray] = None,
    highs: Optional[np.ndarray] = None,
    lows: Optional[np.ndarray] = None,
    spy_prices: Optional[np.ndarray] = None,
    vix_prices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    15-feature matrix. Falls back gracefully if volume/OHLC/market data not available.
    """
    n = len(prices)
    features = np.zeros((n, N_FEATURES), dtype=np.float64)

    if volumes is None:
        volumes = np.ones(n)
    if highs is None:
        highs = prices * 1.005
    if lows is None:
        lows = prices * 0.995
    if spy_prices is None:
        spy_prices = np.full(n, 0.0)
    if vix_prices is None:
        vix_prices = np.full(n, 20.0)

    # Truncate all to same length
    m = min(n, len(volumes), len(highs), len(lows), len(spy_prices), len(vix_prices))
    prices_m = prices[:m]
    spy_m = spy_prices[:m]
    vix_m = vix_prices[:m]

    # 0: normalised price
    p_min, p_max = prices_m.min(), prices_m.max()
    rng = p_max - p_min if p_max != p_min else 1.0
    features[:m, 0] = (prices_m - p_min) / rng

    # 1: daily returns
    rets = np.zeros(m)
    rets[1:] = (prices_m[1:] - prices_m[:-1]) / prices_m[:-1] * 100
    features[:m, 1] = (np.clip(rets, -10, 10) + 10) / 20

    # 2: 5-day z-score
    for i in range(m):
        w = prices_m[max(0, i - 4) : i + 1]
        mu, s = w.mean(), w.std()
        z = (prices_m[i] - mu) / s if s > 0 else 0.0
        features[i, 2] = np.clip(z, -3, 3) / 6 + 0.5

    # 3: sentiment
    features[:m, 3] = float(np.clip((sentiment_score + 1.0) / 2.0, 0.0, 1.0))

    # 4: realised volatility (20-day)
    for i in range(m):
        w = prices_m[max(0, i - 19) : i + 1]
        if len(w) > 1:
            rv = float(np.std(np.diff(w) / w[:-1] * 100))
        else:
            rv = 0.0
        features[i, 4] = float(np.clip(rv / 4.0, 0.0, 1.0))

    # 5: RSI-14
    features[:m, 5] = _rsi(prices_m)[:m]

    # 6: MACD
    features[:m, 6] = _macd(prices_m)[:m]

    # 7: Bollinger position
    features[:m, 7] = _bollinger(prices_m)[:m]

    # 8: SMA-20 distance
    features[:m, 8] = _sma_dist(prices_m)[:m]

    # 9: Rate of Change
    features[:m, 9] = _roc(prices_m)[:m]

    # 10: Volume ratio
    features[:m, 10] = _vol_ratio(volumes[:m])[:m]

    # 11: macro fear (cross-asset volatility proxy)
    for i in range(m):
        w = prices_m[max(0, i - 19) : i + 1]
        if len(w) > 1:
            rv = float(np.std(np.diff(w) / w[:-1] * 100))
        else:
            rv = 0.0
        features[i, 11] = float(np.clip(rv / 6.0, 0.0, 1.0))

    # ─── NEW: Market Context Features ─────────────────────────────────────────

    # 12: SPY 5-day momentum (Are we in a bull/bear market?)
    for i in range(m):
        if i >= 5 and spy_m[i] > 0 and spy_m[i - 5] > 0:
            spy_mom = (spy_m[i] - spy_m[i - 5]) / spy_m[i - 5] * 100
            features[i, 12] = np.clip(spy_mom / 10.0 + 0.5, 0, 1)
        else:
            features[i, 12] = 0.5

    # 13: VIX level (High VIX = fear = bad for stocks)
    # VIX typically ranges 10-40; normalize to [0, 1] with cap at 50
    for i in range(m):
        if vix_m[i] > 0:
            features[i, 13] = np.clip(vix_m[i] / 50.0, 0, 1)
        else:
            features[i, 13] = 0.3  # neutral default

    # 14: Relative strength vs SPY (Is this stock beating the market?)
    for i in range(m):
        if i >= 20 and spy_m[i] > 0 and spy_m[i - 20] > 0 and prices_m[i - 20] > 0:
            stock_ret = (prices_m[i] - prices_m[i - 20]) / prices_m[i - 20] * 100
            spy_ret = (spy_m[i] - spy_m[i - 20]) / spy_m[i - 20] * 100
            rel_strength = stock_ret - spy_ret
            features[i, 14] = np.clip(rel_strength / 20.0 + 0.5, 0, 1)
        else:
            features[i, 14] = 0.5

    return features[:m]


# ─── Classification Targets ───────────────────────────────────────────────────
def _build_classification_targets(prices: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Build multi-horizon classification targets.
    Each target is a bin index (0-3) representing the return magnitude and direction.
    """
    n = len(prices)
    targets = {
        "h1d": np.zeros(n, dtype=np.int64),
        "h3d": np.zeros(n, dtype=np.int64),
        "h7d": np.zeros(n, dtype=np.int64),
    }
    for j in range(n):
        if j + 1 < n and prices[j] > 0:
            ret_1d = (prices[j + 1] - prices[j]) / prices[j] * 100
            targets["h1d"][j] = _classify_return(ret_1d)
        if j + 3 < n and prices[j] > 0:
            ret_3d = (prices[j + 3] - prices[j]) / prices[j] * 100
            targets["h3d"][j] = _classify_return(ret_3d)
        if j + 7 < n and prices[j] > 0:
            ret_7d = (prices[j + 7] - prices[j]) / prices[j] * 100
            targets["h7d"][j] = _classify_return(ret_7d)
    return targets


# ─── Legacy Compatibility ─────────────────────────────────────────────────────
def _build_return_targets(prices: np.ndarray) -> np.ndarray:
    """Kept for backward compatibility with logging. Not used in training."""
    r = np.zeros(len(prices))
    r[1:] = (prices[1:] - prices[:-1]) / prices[:-1] * 100
    return (np.clip(r, -10, 10) + 10) / 20


# ─── Sequence Building ───────────────────────────────────────────────────────
def _make_sequences(
    features: np.ndarray, targets: Dict[str, np.ndarray]
) -> tuple:
    """
    Build (X, y_dict) from feature matrix and classification targets.
    The anchor day is the last day in each sequence window.
    Only includes samples where ALL horizons (including 7d) are valid.
    """
    n = len(features)
    # Anchor day j = i + SEQ_LEN - 1; need j + 7 < len(prices) for 7d target.
    # Since targets are indexed the same as prices, limit is n - SEQ_LEN - 6
    limit = min(n - SEQ_LEN, n - SEQ_LEN - 6)
    if limit <= 0:
        # Not enough data — return empty tensors
        empty_x = torch.zeros((0, SEQ_LEN, features.shape[1]), dtype=torch.float32)
        empty_y = {
            "h1d": torch.zeros(0, dtype=torch.long),
            "h3d": torch.zeros(0, dtype=torch.long),
            "h7d": torch.zeros(0, dtype=torch.long),
        }
        return empty_x, empty_y

    X, y1d, y3d, y7d = [], [], [], []
    for i in range(limit):
        X.append(features[i : i + SEQ_LEN])
        j = i + SEQ_LEN - 1  # anchor day (last day in window)
        y1d.append(targets["h1d"][j])
        y3d.append(targets["h3d"][j])
        y7d.append(targets["h7d"][j])

    return (
        torch.tensor(np.array(X), dtype=torch.float32),
        {
            "h1d": torch.tensor(np.array(y1d), dtype=torch.long),
            "h3d": torch.tensor(np.array(y3d), dtype=torch.long),
            "h7d": torch.tensor(np.array(y7d), dtype=torch.long),
        },
    )


def _get_trading_dates(last_date: datetime, n: int) -> List[str]:
    dates, d = [], last_date
    while len(dates) < n:
        d += timedelta(days=1)
        if d.weekday() < 5:
            dates.append(d.strftime("%Y-%m-%d"))
    return dates


# ─── Data Fetch ──────────────────────────────────────────────────────────────
def _fetch_ohlcv(ticker: str) -> dict:
    try:
        stock = yf.Ticker(ticker.upper())
        hist = stock.history(period=HISTORY_PERIOD, interval="1d", auto_adjust=True)
    except Exception as e:
        raise ValueError(f"Failed to fetch data for {ticker.upper()}: {e}")
    if hist is None or hist.empty or "Close" not in hist.columns:
        raise ValueError(f"No price data for {ticker.upper()}.")
    closes = hist["Close"].dropna().values.astype(np.float64)
    volumes = (
        hist["Volume"].values.astype(np.float64)
        if "Volume" in hist.columns
        else np.ones(len(closes))
    )
    highs = (
        hist["High"].values.astype(np.float64)
        if "High" in hist.columns
        else closes * 1.005
    )
    lows = (
        hist["Low"].values.astype(np.float64)
        if "Low" in hist.columns
        else closes * 0.995
    )
    n = min(len(closes), len(volumes), len(highs), len(lows))
    if n < SEQ_LEN + 10:
        raise ValueError(f"Insufficient history for {ticker.upper()}.")

    # ─── Fetch Market Context (SPY + VIX) ─────────────────────────────────────
    spy_prices = np.full(n, 0.0, dtype=np.float64)
    vix_prices = np.full(n, 20.0, dtype=np.float64)

    try:
        stock_dates = hist.index.normalize()[:n]

        # SPY (S&P 500 proxy)
        spy_hist = yf.Ticker("SPY").history(
            period=HISTORY_PERIOD, interval="1d", auto_adjust=True
        )
        if spy_hist is not None and not spy_hist.empty and "Close" in spy_hist.columns:
            spy_by_date = dict(
                zip(spy_hist.index.normalize(), spy_hist["Close"].values)
            )
            for i, d in enumerate(stock_dates):
                if d in spy_by_date:
                    spy_prices[i] = float(spy_by_date[d])

        # VIX (CBOE Volatility Index)
        vix_hist = yf.Ticker("^VIX").history(
            period=HISTORY_PERIOD, interval="1d", auto_adjust=True
        )
        if vix_hist is not None and not vix_hist.empty and "Close" in vix_hist.columns:
            vix_by_date = dict(
                zip(vix_hist.index.normalize(), vix_hist["Close"].values)
            )
            for i, d in enumerate(stock_dates):
                if d in vix_by_date:
                    vix_prices[i] = float(vix_by_date[d])
    except Exception:
        pass  # graceful degradation — market features will be neutral defaults

    # Forward-fill any zero gaps in SPY/VIX (weekends already excluded by yfinance)
    for arr in [spy_prices, vix_prices]:
        last_val = arr[0] if arr[0] > 0 else (500.0 if arr is spy_prices else 20.0)
        for i in range(len(arr)):
            if arr[i] <= 0:
                arr[i] = last_val
            else:
                last_val = arr[i]

    return {
        "closes": closes[:n],
        "volumes": volumes[:n],
        "highs": highs[:n],
        "lows": lows[:n],
        "hist": hist,
        "spy_prices": spy_prices,
        "vix_prices": vix_prices,
    }


# ─── Training ─────────────────────────────────────────────────────────────────
def _train_lstm(
    X: torch.Tensor, y: Dict[str, torch.Tensor], seed: int, lr_mult: float = 1.0
) -> LSTMPredictor:
    torch.manual_seed(seed)
    model = LSTMPredictor()
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=LSTM_LR * lr_mult, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=LSTM_EPOCHS, eta_min=1e-5
    )
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(LSTM_EPOCHS):
        optimiser.zero_grad()
        outputs = model(X)
        loss = (
            criterion(outputs["h1d"], y["h1d"])
            + criterion(outputs["h3d"], y["h3d"])
            + criterion(outputs["h7d"], y["h7d"])
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        scheduler.step()
    return model


def _train_transformer(
    X: torch.Tensor, y: Dict[str, torch.Tensor], seed: int
) -> TransformerPredictor:
    torch.manual_seed(seed + 9999)
    model = TransformerPredictor()
    optimiser = torch.optim.AdamW(model.parameters(), lr=TF_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=TF_EPOCHS, eta_min=1e-5
    )
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(TF_EPOCHS):
        optimiser.zero_grad()
        outputs = model(X)
        loss = (
            criterion(outputs["h1d"], y["h1d"])
            + criterion(outputs["h3d"], y["h3d"])
            + criterion(outputs["h7d"], y["h7d"])
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        scheduler.step()
    return model


def _train_xgboost(
    X: torch.Tensor, y: Dict[str, torch.Tensor]
) -> Optional[Dict[str, Any]]:
    """Train 3 XGBoost classifiers (one per horizon). Returns None if XGBoost unavailable."""
    if not HAS_XGBOOST:
        return None

    # Flatten 3D sequences (batch, seq_len, features) → 2D (batch, seq_len*features)
    X_flat = X.numpy().reshape(X.shape[0], -1)

    models = {}
    for h_key in ["h1d", "h3d", "h7d"]:
        y_arr = y[h_key].numpy()
        try:
            clf = xgb.XGBClassifier(
                n_estimators=50,   # reduced from 100 for speed
                max_depth=4,
                learning_rate=0.1,  # higher LR with fewer trees
                subsample=0.8,
                objective="multi:softprob",
                num_class=N_BINS,
                eval_metric="mlogloss",
                verbosity=0,
                use_label_encoder=False,
            )
            clf.fit(X_flat, y_arr)
            models[h_key] = clf
        except Exception:
            pass  # skip this horizon if XGBoost fails

    return models if models else None


# ─── Direct Multi-Horizon Prediction (NO autoregression) ─────────────────────
def _direct_predict(
    nn_models: list,
    xgb_models: Optional[Dict[str, Any]],
    features: np.ndarray,
    prices: np.ndarray,
) -> Dict[str, Any]:
    """
    Predict Day 1, Day 3, Day 7 directly from the last SEQ_LEN real data points.
    No autoregressive rollout — each prediction uses only real, observed data.

    Returns dict with keys 'h1d', 'h3d', 'h7d', each containing:
      - probs: probability distribution over 4 bins
      - expected_return: weighted average return (%)
      - predicted_price: last_price * (1 + return/100)
      - direction: 'rise' or 'fall'
      - confidence_raw: max probability (model conviction)
    """
    seq = torch.tensor(features[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0)
    last_price = float(prices[-1])

    # Collect probability distributions from all models
    all_probs: Dict[str, list] = {"h1d": [], "h3d": [], "h7d": []}

    # Neural network predictions
    for m in nn_models:
        m.eval()
        with torch.no_grad():
            out = m(seq)
            for h_key in ["h1d", "h3d", "h7d"]:
                probs = torch.softmax(out[h_key], dim=-1).squeeze(0).numpy()
                all_probs[h_key].append(probs)

    # XGBoost predictions
    if xgb_models:
        X_flat = features[-SEQ_LEN:].reshape(1, -1)
        for h_key in ["h1d", "h3d", "h7d"]:
            if h_key in xgb_models:
                try:
                    xgb_probs = xgb_models[h_key].predict_proba(X_flat)[0]
                    if len(xgb_probs) == N_BINS:
                        all_probs[h_key].append(xgb_probs)
                except Exception:
                    pass

    # Average probabilities across all models (neural nets weighted vs XGBoost)
    results = {}
    for h_key in ["h1d", "h3d", "h7d"]:
        probs_list = all_probs[h_key]
        if not probs_list:
            # Fallback: uniform distribution
            avg_probs = np.array([0.25, 0.25, 0.25, 0.25])
        else:
            # Separate NN and XGB probs for weighted blending
            nn_probs_list = []
            xgb_probs_list = []
            for i, p in enumerate(probs_list):
                if xgb_models and h_key in xgb_models and i == len(probs_list) - 1:
                    xgb_probs_list.append(p)
                else:
                    nn_probs_list.append(p)

            if nn_probs_list and xgb_probs_list:
                nn_avg = np.mean(nn_probs_list, axis=0)
                xgb_avg = np.mean(xgb_probs_list, axis=0)
                avg_probs = NN_WEIGHT * nn_avg + XGB_WEIGHT * xgb_avg
            else:
                avg_probs = np.mean(probs_list, axis=0)

        # Expected return = sum(prob_i * midpoint_i)
        expected_return = float(np.sum(avg_probs * BIN_MIDPOINTS))
        predicted_price = last_price * (1 + expected_return / 100)

        results[h_key] = {
            "probs": avg_probs.tolist(),
            "expected_return": round(expected_return, 4),
            "predicted_price": round(predicted_price, 4),
            "direction": "rise" if expected_return >= 0 else "fall",
            "confidence_raw": float(np.max(avg_probs)),
        }

    return results


# ─── Price Interpolation ─────────────────────────────────────────────────────
def _interpolate_prices(
    last_price: float,
    day1_price: float,
    day3_price: float,
    day7_price: float,
    hist_vol: float = 1.0,
    seed: int = 42,
) -> List[float]:
    """
    Interpolate 7 daily prices from 3 anchor points (Day 1, 3, 7).
    Adds small volatility-scaled jitter to intermediate days so the chart
    looks realistic (not a straight line between anchors).
    """
    rng = np.random.RandomState(seed)

    # Anchor points: day 0 = last_price (known), day 1/3/7 = predicted
    anchors_x = [0, 1, 3, 7]
    anchors_y = [last_price, day1_price, day3_price, day7_price]

    prices = []
    for day in range(1, 8):
        # Find surrounding anchors and linearly interpolate
        for k in range(len(anchors_x) - 1):
            if anchors_x[k] <= day <= anchors_x[k + 1]:
                t = (day - anchors_x[k]) / (anchors_x[k + 1] - anchors_x[k])
                p = anchors_y[k] + t * (anchors_y[k + 1] - anchors_y[k])
                break
        else:
            p = day7_price  # shouldn't happen

        # Add small jitter to non-anchor days for visual realism
        if day not in [1, 3, 7]:
            jitter_scale = hist_vol * 0.002 * last_price  # tiny, proportional noise
            p += rng.normal(0, jitter_scale)

        prices.append(round(float(max(p, 0.01)), 4))

    return prices


# ─── Signals & Reasoning ─────────────────────────────────────────────────────
def _compute_signals(prices: np.ndarray, hist_prices: list, hist_dates: list) -> dict:
    closes = np.array(hist_prices, dtype=np.float64)
    current = closes[-1]
    signals = {}
    if len(closes) >= 20:
        ma20 = float(np.mean(closes[-20:]))
        signals["MA20"] = {"value": round(ma20, 2), "label": "20-Day Moving Avg"}
    if len(closes) >= 50:
        ma50 = float(np.mean(closes[-50:]))
        signals["MA50"] = {"value": round(ma50, 2), "label": "50-Day Moving Avg"}
    if len(closes) >= 6:
        mom5 = ((current - closes[-6]) / closes[-6]) * 100
        signals["Momentum5D"] = {"value": round(mom5, 2), "label": "5-Day Momentum (%)"}
    if len(closes) >= 30:
        vol = float(np.std(closes[-30:]) / np.mean(closes[-30:]) * 100)
        signals["Volatility30D"] = {
            "value": round(vol, 2),
            "label": "30-Day Volatility (%)",
        }
    if len(closes) >= 15:
        deltas = np.diff(closes[-15:])
        gains = np.mean(deltas[deltas > 0]) if (deltas > 0).any() else 0.0
        losses = np.mean(-deltas[deltas < 0]) if (deltas < 0).any() else 0.0
        rs = gains / losses if losses != 0 else 100
        signals["RSI14"] = {
            "value": round(100 - 100 / (1 + rs), 1),
            "label": "RSI (14-period)",
        }
    return signals


def _build_reasoning(
    direction: str,
    pct_change: float,
    confidence: float,
    signals: dict,
    sentiment_score: float = 0.0,
    horizon_results: Optional[Dict] = None,
) -> list[str]:
    reasons = [
        f"AI model forecasts a {direction.upper()} of {pct_change:.2f}% "
        f"over the next 7 trading days (confidence: {confidence}%)."
    ]

    # Multi-horizon breakdown
    if horizon_results:
        h1d = horizon_results.get("h1d", {})
        h3d = horizon_results.get("h3d", {})
        h7d = horizon_results.get("h7d", {})
        reasons.append(
            f"Direct predictions: Day 1 {h1d.get('expected_return', 0):+.2f}%, "
            f"Day 3 {h3d.get('expected_return', 0):+.2f}%, "
            f"Day 7 {h7d.get('expected_return', 0):+.2f}%."
        )

    ma20 = signals.get("MA20", {}).get("value")
    ma50 = signals.get("MA50", {}).get("value")
    if ma20 and ma50:
        if ma20 > ma50:
            reasons.append(
                f"MA20 (${ma20:.2f}) above MA50 (${ma50:.2f}) — bullish momentum."
            )
        else:
            reasons.append(
                f"MA20 (${ma20:.2f}) below MA50 (${ma50:.2f}) — bearish pressure."
            )
    mom = signals.get("Momentum5D", {}).get("value")
    if mom is not None:
        reasons.append(
            f"5-day momentum {'positive' if mom > 0 else 'negative'} at {mom:+.2f}%."
        )
    rsi = signals.get("RSI14", {}).get("value")
    if rsi is not None:
        if rsi > 70:
            reasons.append(f"RSI = {rsi:.1f} — overbought, potential pullback.")
        elif rsi < 30:
            reasons.append(f"RSI = {rsi:.1f} — oversold, potential rebound.")
        else:
            reasons.append(f"RSI = {rsi:.1f} — neutral zone.")
    if sentiment_score > 0.1:
        reasons.append(
            f"News sentiment positive ({sentiment_score:+.2f}) — bullish signal."
        )
    elif sentiment_score < -0.1:
        reasons.append(
            f"News sentiment negative ({sentiment_score:+.2f}) — bearish signal."
        )
    vol = signals.get("Volatility30D", {}).get("value")
    if vol is not None:
        if vol > 5:
            reasons.append(f"30-day volatility elevated at {vol:.2f}%.")
        else:
            reasons.append(f"30-day volatility low at {vol:.2f}%.")
    return reasons


# ─── Prediction Log ───────────────────────────────────────────────────────────
def _load_log() -> dict:
    if os.path.exists(LOG_PATH):
        try:
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_log(log: dict) -> None:
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)


def save_prediction(
    ticker: str,
    predictions: List[dict],
    direction: str,
    confidence: float,
    made_at: str,
) -> None:
    log = _load_log()
    if ticker not in log:
        log[ticker] = []
    log[ticker].append(
        {
            "made_at": made_at,
            "direction": direction,
            "confidence": confidence,
            "predictions": predictions,
            "outcomes": [],
        }
    )
    _save_log(log)


def check_past_predictions(ticker: str) -> List[dict]:
    log = _load_log()
    entries = log.get(ticker, [])
    if not entries:
        return []
    today = datetime.utcnow().date()
    changed = False
    try:
        stock = yf.Ticker(ticker.upper())
        hist = stock.history(period="1y", interval="1d", auto_adjust=True)
        prices_by_date = {
            d.strftime("%Y-%m-%d"): round(float(v), 4)
            for d, v in zip(hist.index, hist["Close"].dropna().values)
        }
    except Exception:
        prices_by_date = {}
    for entry in entries:
        existing = {o["date"] for o in entry.get("outcomes", [])}
        for pred in entry.get("predictions", []):
            pd_ = pred["date"]
            if pd_ in existing:
                continue
            try:
                pd_date = datetime.strptime(pd_, "%Y-%m-%d").date()
            except ValueError:
                continue
            if pd_date > today:
                continue
            actual = prices_by_date.get(pd_)
            if actual is None:
                for off in range(1, 4):
                    d2 = (pd_date + timedelta(days=off)).strftime("%Y-%m-%d")
                    if d2 in prices_by_date:
                        actual = prices_by_date[d2]
                        break
            if actual is not None:
                predicted = pred["price"]
                error_pct = round(((actual - predicted) / predicted) * 100, 2)
                correct = (actual > predicted and entry["direction"] == "rise") or (
                    actual < predicted and entry["direction"] == "fall"
                )
                entry.setdefault("outcomes", []).append(
                    {
                        "date": pd_,
                        "predicted": predicted,
                        "actual": actual,
                        "error_pct": error_pct,
                        "correct": correct,
                    }
                )
                changed = True
    if changed:
        log[ticker] = entries
        _save_log(log)
    return entries


def get_prediction_history(ticker: str) -> List[dict]:
    return check_past_predictions(ticker)


# ─── Fine-Tuning (BPTT) ──────────────────────────────────────────────────────
FT_EPOCHS = 10
FT_LR = 0.001
FT_LOG_KEY = "used_outcomes"


def fine_tune_on_outcomes(
    ticker: str, prices: np.ndarray, all_dates: List[str]
) -> Dict[str, Any]:
    cache_path = os.path.join(MODEL_CACHE_DIR, f"{ticker.upper()}.pt")
    meta_path = os.path.join(MODEL_CACHE_DIR, f"{ticker.upper()}.json")
    ft_log_path = os.path.join(MODEL_CACHE_DIR, f"{ticker.upper()}_ftlog.json")

    if not os.path.exists(cache_path):
        return {"fine_tuned": False, "reason": "no_cached_model"}

    ft_used: set = set()
    ft_count_total = 0
    if os.path.exists(ft_log_path):
        try:
            with open(ft_log_path, "r", encoding="utf-8") as f:
                d = json.load(f)
            ft_used = set(d.get(FT_LOG_KEY, []))
            ft_count_total = d.get("total_fine_tunes", 0)
        except Exception:
            pass

    log = _load_log()
    entries = log.get(ticker.upper(), [])
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    sent = get_sentiment(ticker.upper()) or 0.0
    features = _build_feature_matrix(prices, sentiment_score=sent)
    targets = _build_classification_targets(prices)

    new_samples = []
    for entry in entries:
        for outcome in entry.get("outcomes", []):
            key = f"{outcome['date']}|{outcome['predicted']:.4f}"
            if key in ft_used:
                continue
            idx = date_to_idx.get(outcome["date"])
            if idx is None or idx < SEQ_LEN:
                continue
            new_samples.append(
                {
                    "key": key,
                    "idx": idx,
                    "actual": outcome["actual"],
                    "predicted": outcome["predicted"],
                }
            )

    if not new_samples:
        return {
            "fine_tuned": False,
            "reason": "no_new_outcomes",
            "total_fine_tunes": ft_count_total,
        }

    # Fine-tune the cached LSTM model
    model = LSTMPredictor()
    try:
        model.load_state_dict(
            torch.load(cache_path, map_location="cpu", weights_only=True)
        )
    except Exception as e:
        # Architecture changed — old cache incompatible, skip fine-tuning
        return {"fine_tuned": False, "reason": f"architecture_changed: {e}"}

    optimiser = torch.optim.Adam(model.parameters(), lr=FT_LR)
    criterion = nn.CrossEntropyLoss()
    pre_losses, post_losses, used_keys = [], [], []

    for sample in new_samples:
        idx = sample["idx"]
        if idx < SEQ_LEN or idx >= len(features):
            continue

        # Get classification target for this index
        target_dict = {
            "h1d": torch.tensor([targets["h1d"][idx]], dtype=torch.long),
            "h3d": torch.tensor([targets["h3d"][idx]], dtype=torch.long),
            "h7d": torch.tensor([targets["h7d"][idx]], dtype=torch.long),
        }
        seq = torch.tensor(
            features[idx - SEQ_LEN : idx], dtype=torch.float32
        ).unsqueeze(0)

        model.eval()
        with torch.no_grad():
            out = model(seq)
            pre_loss = sum(
                criterion(out[h], target_dict[h]).item()
                for h in ["h1d", "h3d", "h7d"]
            )
            pre_losses.append(pre_loss)

        model.train()
        for _ in range(FT_EPOCHS):
            optimiser.zero_grad()
            out = model(seq)
            loss = sum(
                criterion(out[h], target_dict[h])
                for h in ["h1d", "h3d", "h7d"]
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

        model.eval()
        with torch.no_grad():
            out = model(seq)
            post_loss = sum(
                criterion(out[h], target_dict[h]).item()
                for h in ["h1d", "h3d", "h7d"]
            )
            post_losses.append(post_loss)
        used_keys.append(sample["key"])

    if not used_keys:
        return {
            "fine_tuned": False,
            "reason": "no_valid_samples",
            "total_fine_tunes": ft_count_total,
        }

    try:
        torch.save(model.state_dict(), cache_path)
        with open(meta_path, "w", encoding="utf-8") as mf:
            json.dump(
                {
                    "saved_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
                    "ticker": ticker.upper(),
                    "fine_tuned": True,
                    "outcomes_used": ft_count_total + len(used_keys),
                    "version": "v4_classification",
                },
                mf,
            )
    except Exception as e:
        return {"fine_tuned": False, "reason": f"save_error: {e}"}

    ft_used.update(used_keys)
    ft_count_total += len(used_keys)
    try:
        with open(ft_log_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    FT_LOG_KEY: list(ft_used),
                    "total_fine_tunes": ft_count_total,
                    "last_ft_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
                },
                f,
                indent=2,
            )
    except Exception:
        pass

    avg_pre = round(float(np.mean(pre_losses)), 6) if pre_losses else 0
    avg_post = round(float(np.mean(post_losses)), 6) if post_losses else 0
    improv = round((avg_pre - avg_post) / avg_pre * 100, 1) if avg_pre > 0 else 0
    return {
        "fine_tuned": True,
        "outcomes_trained": len(used_keys),
        "total_fine_tunes": ft_count_total,
        "avg_loss_before": avg_pre,
        "avg_loss_after": avg_post,
        "loss_improvement_pct": improv,
        "last_ft_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    }


# ─── Auto-Retrain: Invalidate Cache if Accuracy < 50% ────────────────────────
ACCURACY_THRESHOLD = 50.0   # % win rate below which we force a full retrain
MIN_OUTCOMES_FOR_CHECK = 5  # need at least this many resolved predictions

def _check_accuracy_and_maybe_retrain(ticker: str) -> Dict[str, Any]:
    """
    Compute the direction win rate across ALL resolved past predictions for
    a ticker. If win rate < ACCURACY_THRESHOLD and we have enough samples,
    delete the model cache so the next predict_stock call re-trains from
    scratch (rather than loading a stale, degraded model).

    Returns a dict with the accuracy check results.
    """
    log = _load_log()
    entries = log.get(ticker.upper(), [])

    # Collect all resolved outcomes
    outcomes = []
    for entry in entries:
        direction = entry.get("direction", "")
        for o in entry.get("outcomes", []):
            if "correct" in o:
                outcomes.append(o["correct"])

    if len(outcomes) < MIN_OUTCOMES_FOR_CHECK:
        return {
            "accuracy_check": False,
            "reason": f"only {len(outcomes)} resolved outcomes (need {MIN_OUTCOMES_FOR_CHECK})",
        }

    wins = sum(1 for c in outcomes if c)
    win_rate = round(wins / len(outcomes) * 100, 1)

    cache_path = os.path.join(MODEL_CACHE_DIR, f"{ticker.upper()}.pt")
    meta_path  = os.path.join(MODEL_CACHE_DIR, f"{ticker.upper()}.json")

    if win_rate < ACCURACY_THRESHOLD:
        # Invalidate model cache to force full retrain next call
        deleted = []
        for p in [cache_path, meta_path]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                    deleted.append(os.path.basename(p))
                except Exception:
                    pass
        return {
            "accuracy_check": True,
            "win_rate": win_rate,
            "total_outcomes": len(outcomes),
            "cache_invalidated": len(deleted) > 0,
            "deleted_files": deleted,
            "reason": (
                f"Win rate {win_rate}% < threshold {ACCURACY_THRESHOLD}% "
                f"— cache purged, full retrain scheduled."
            ),
        }

    return {
        "accuracy_check": True,
        "win_rate": win_rate,
        "total_outcomes": len(outcomes),
        "cache_invalidated": False,
        "reason": f"Win rate {win_rate}% ≥ threshold — model healthy.",
    }


# ─── Backtest ─────────────────────────────────────────────────────────────────
def backtest_stock(ticker: str, days_back: int = 7) -> Dict[str, Any]:
    try:
        ohlcv = _fetch_ohlcv(ticker)
    except Exception as exc:
        return {"ticker": ticker.upper(), "error": str(exc)}

    try:
        all_prices = ohlcv["closes"]
        all_vols = ohlcv["volumes"]
        all_highs = ohlcv["highs"]
        all_lows = ohlcv["lows"]
        spy_prices = ohlcv["spy_prices"]
        vix_prices = ohlcv["vix_prices"]
        hist = ohlcv["hist"]
        all_dates = [d.strftime("%Y-%m-%d") for d in hist.index][: len(all_prices)]

        if len(all_prices) < SEQ_LEN + days_back + 10:
            return {"ticker": ticker.upper(), "error": "Insufficient history."}

        cut_idx = len(all_prices) - days_back
        train_prices = all_prices[:cut_idx]
        actual_prices = all_prices[cut_idx:]
        actual_dates = all_dates[cut_idx:]
        cutoff_date = all_dates[cut_idx - 1]

        cached_sent = get_sentiment(ticker.upper()) or 0.0
        train_features = _build_feature_matrix(
            train_prices,
            sentiment_score=cached_sent,
            volumes=all_vols[:cut_idx],
            highs=all_highs[:cut_idx],
            lows=all_lows[:cut_idx],
            spy_prices=spy_prices[:cut_idx],
            vix_prices=vix_prices[:cut_idx],
        )
        targets = _build_classification_targets(train_prices)
        X, y = _make_sequences(train_features, targets)

        if X.shape[0] == 0:
            return {"ticker": ticker.upper(), "error": "Insufficient training data."}

        seed = sum(ord(c) * (i + 1) for i, c in enumerate(ticker.upper())) % (2**31)
        np.random.seed(seed % (2**31))

        # Train ensemble
        nn_models = [
            _train_lstm(X, y, seed + i, [0.8, 1.2][i % 2]) for i in range(N_LSTM)
        ] + [_train_transformer(X, y, seed)]

        xgb_models = _train_xgboost(X, y)

        # Direct multi-horizon prediction
        horizon_results = _direct_predict(
            nn_models, xgb_models, train_features, train_prices
        )

        # Compute historical volatility for interpolation
        hist_rets = np.diff(train_prices) / train_prices[:-1] * 100
        hist_vol = float(np.std(hist_rets[-30:])) if len(hist_rets) >= 30 else 1.0

        # Interpolate daily prices
        last_train = float(train_prices[-1])
        predicted_prices = _interpolate_prices(
            last_train,
            horizon_results["h1d"]["predicted_price"],
            horizon_results["h3d"]["predicted_price"],
            horizon_results["h7d"]["predicted_price"],
            hist_vol=hist_vol,
            seed=seed,
        )

        # Trim to match actual days available
        n_compare = min(len(predicted_prices), len(actual_prices))
        predicted_prices = predicted_prices[:n_compare]

        direction = "rise" if predicted_prices[-1] > last_train else "fall"
        actual_dir = (
            "rise" if float(actual_prices[n_compare - 1]) > last_train else "fall"
        )

        rows, hits, total_err = [], 0, 0.0
        for i in range(n_compare):
            pred_p = predicted_prices[i]
            act_p = round(float(actual_prices[i]), 4)
            error = (
                round(((act_p - pred_p) / pred_p) * 100, 2) if pred_p != 0 else 0.0
            )
            prev_pred = predicted_prices[i - 1] if i > 0 else last_train
            prev_act = float(actual_prices[i - 1]) if i > 0 else last_train
            hit = ("rise" if pred_p > prev_pred else "fall") == (
                "rise" if act_p > prev_act else "fall"
            )
            if hit:
                hits += 1
            total_err += abs(error)
            rows.append(
                {
                    "date": actual_dates[i] if i < len(actual_dates) else "unknown",
                    "predicted": pred_p,
                    "actual": act_p,
                    "error_pct": error,
                    "hit": hit,
                }
            )

        n = len(rows)
        d_acc = round(hits / n * 100, 1) if n > 0 else 0
        a_err = round(total_err / n, 2) if n > 0 else 0
        grade = (
            "A"
            if d_acc >= 70 and a_err < 5
            else "B" if d_acc >= 60 or a_err < 8 else "C" if d_acc >= 50 else "D"
        )

        return {
            "ticker": ticker.upper(),
            "days_back": days_back,
            "cutoff_date": cutoff_date,
            "direction_accuracy": d_acc,
            "avg_price_error": a_err,
            "overall_direction": direction,
            "actual_direction": actual_dir,
            "direction_correct": direction == actual_dir,
            "grade": grade,
            "rows": rows,
            "horizon_predictions": {
                h: {
                    "expected_return": horizon_results[h]["expected_return"],
                    "direction": horizon_results[h]["direction"],
                }
                for h in ["h1d", "h3d", "h7d"]
            },
        }

    except Exception as exc:
        return {"ticker": ticker.upper(), "error": str(exc)}


# ─── Main Predict ─────────────────────────────────────────────────────────────
def predict_stock(ticker: str) -> Dict[str, Any]:
    """
    Multi-Horizon Classification Ensemble prediction.
    2 LSTMs + 1 Transformer + XGBoost, all predicting 3 horizons (1d/3d/7d).
    15 features including RSI, MACD, Bollinger, volume, sentiment, SPY, VIX.
    No autoregressive rollout — each horizon predicted directly from real data.
    """
    try:
        ohlcv = _fetch_ohlcv(ticker)
    except Exception as exc:
        return {"ticker": ticker.upper(), "error": str(exc)}

    prices = ohlcv["closes"]
    volumes = ohlcv["volumes"]
    highs = ohlcv["highs"]
    lows = ohlcv["lows"]
    spy_prices = ohlcv["spy_prices"]
    vix_prices = ohlcv["vix_prices"]

    sentiment_score = get_sentiment(ticker.upper()) or 0.0
    sentiment_used = sentiment_score != 0.0

    features = _build_feature_matrix(
        prices,
        sentiment_score=sentiment_score,
        volumes=volumes,
        highs=highs,
        lows=lows,
        spy_prices=spy_prices,
        vix_prices=vix_prices,
    )
    targets = _build_classification_targets(prices)
    X, y = _make_sequences(features, targets)

    if X.shape[0] == 0:
        return {"ticker": ticker.upper(), "error": "Insufficient data for training."}

    seed = sum(ord(c) * (i + 1) for i, c in enumerate(ticker.upper())) % (2**31)
    np.random.seed(seed % (2**31))

    try:
        check_past_predictions(ticker.upper())
        _check_accuracy_and_maybe_retrain(ticker.upper())
    except Exception:
        pass

    ft_info: Dict = {"fine_tuned": False}
    try:
        stock_ft = yf.Ticker(ticker.upper())
        hist_ft = stock_ft.history(period="1y", interval="1d", auto_adjust=True)
        if not hist_ft.empty and "Close" in hist_ft.columns:
            ft_dates = [d.strftime("%Y-%m-%d") for d in hist_ft.index]
            ft_prices = hist_ft["Close"].dropna().values.astype(np.float64)
            ft_info = fine_tune_on_outcomes(ticker.upper(), ft_prices, ft_dates)
    except Exception:
        pass

    # Cache: save/load first LSTM only (Transformer + XGBoost train fast anyway)
    cache_path = os.path.join(MODEL_CACHE_DIR, f"{ticker.upper()}.pt")
    meta_path = os.path.join(MODEL_CACHE_DIR, f"{ticker.upper()}.json")
    cache_valid = False
    cached_lstm = LSTMPredictor()

    if os.path.exists(cache_path) and os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as mf:
                meta = json.load(mf)
            # Only use cache if it's the v4 classification architecture
            if meta.get("version") == "v4_classification":
                saved_at = datetime.strptime(meta["saved_at"], "%Y-%m-%d %H:%M UTC")
                if (datetime.utcnow() - saved_at).total_seconds() / 3600 < CACHE_HOURS:
                    cached_lstm.load_state_dict(
                        torch.load(cache_path, map_location="cpu", weights_only=True)
                    )
                    cache_valid = True
        except Exception:
            cache_valid = False

    # Build ensemble: 2 LSTMs + 1 Transformer + XGBoost
    if cache_valid:
        lstm1 = cached_lstm
        lstm2 = _train_lstm(X, y, seed + 1, 1.2)
    else:
        lstm1 = _train_lstm(X, y, seed, 0.8)
        lstm2 = _train_lstm(X, y, seed + 1, 1.2)
        try:
            torch.save(lstm1.state_dict(), cache_path)
            with open(meta_path, "w") as mf:
                json.dump(
                    {
                        "saved_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
                        "ticker": ticker.upper(),
                        "version": "v4_classification",
                    },
                    mf,
                )
        except Exception:
            pass

    transformer = _train_transformer(X, y, seed)
    nn_models = [lstm1, lstm2, transformer]

    # Train XGBoost ensemble member
    xgb_models = _train_xgboost(X, y)

    # Direct multi-horizon prediction (NO autoregressive rollout)
    horizon_results = _direct_predict(nn_models, xgb_models, features, prices)

    # Compute historical volatility for interpolation jitter
    hist_rets = np.diff(prices) / prices[:-1] * 100
    hist_vol = float(np.std(hist_rets[-30:])) if len(hist_rets) >= 30 else 1.0

    # Interpolate 7 daily prices from the 3 anchor points
    last_price_val = float(prices[-1])
    future_prices = _interpolate_prices(
        last_price_val,
        horizon_results["h1d"]["predicted_price"],
        horizon_results["h3d"]["predicted_price"],
        horizon_results["h7d"]["predicted_price"],
        hist_vol=hist_vol,
        seed=seed,
    )

    # Fetch display history
    try:
        sh = yf.Ticker(ticker.upper()).history(
            period="3mo", interval="1d", auto_adjust=True
        )
        if sh is None or sh.empty:
            raise ValueError("empty")
        hist_dates = [d.strftime("%Y-%m-%d") for d in sh.index]
        hist_prices = [round(float(v), 4) for v in sh["Close"].dropna().values]
        last_date = sh.index[-1].to_pydatetime()
    except Exception:
        hist_prices = [round(float(p), 4) for p in prices[-90:]]
        today = datetime.utcnow()
        hist_dates, d = [], today
        for _ in hist_prices:
            while d.weekday() >= 5:
                d -= timedelta(days=1)
            hist_dates.insert(0, d.strftime("%Y-%m-%d"))
            d -= timedelta(days=1)
        last_date = datetime.strptime(hist_dates[-1], "%Y-%m-%d")

    pred_dates = _get_trading_dates(last_date, PREDICT_DAYS)

    # Direction & confidence from multi-horizon ensemble
    last_hist = hist_prices[-1]
    day7_ret = horizon_results["h7d"]["expected_return"]
    day1_ret = horizon_results["h1d"]["expected_return"]
    weighted = day1_ret * 0.4 + day7_ret * 0.6
    direction = "rise" if weighted >= 0 else "fall"
    pct_change = abs(day7_ret)

    # Confidence from average model conviction across horizons
    avg_conviction = np.mean([
        horizon_results[h]["confidence_raw"] for h in ["h1d", "h3d", "h7d"]
    ])
    # Map conviction (0.25 = random, 1.0 = certain) to display confidence (50-95%)
    confidence = round(50 + 45 * min(1.0, (avg_conviction - 0.25) / 0.5), 1)
    confidence = max(50.0, min(95.0, confidence))

    signals = _compute_signals(prices, hist_prices, hist_dates)
    reasoning = _build_reasoning(
        direction, pct_change, confidence, signals, sentiment_score, horizon_results
    )

    predictions_payload = [
        {"date": d, "price": p} for d, p in zip(pred_dates, future_prices)
    ]

    made_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    try:
        save_prediction(
            ticker.upper(), predictions_payload, direction, confidence, made_at
        )
    except Exception:
        pass

    return {
        "ticker": ticker.upper(),
        "historical": [
            {"date": d, "price": p} for d, p in zip(hist_dates, hist_prices)
        ],
        "predictions": predictions_payload,
        "direction": direction,
        "confidence": confidence,
        "signals": signals,
        "reasoning": reasoning,
        "made_at": made_at,
        "online_learning": ft_info,
        "sentiment_feature": {
            "score": round(sentiment_score, 4),
            "used": sentiment_used,
            "label": (
                "bullish"
                if sentiment_score > 0.1
                else "bearish" if sentiment_score < -0.1 else "neutral"
            ),
            "note": (
                "News sentiment from AI research injected as model feature"
                if sentiment_used
                else "Run AI Research first to enable sentiment-enhanced prediction"
            ),
        },
        "model_info": {
            "architecture": "Multi-Horizon Classification Ensemble (v4)",
            "models": f"{N_LSTM} LSTM + {N_TRANSFORM} Transformer"
            + (" + XGBoost" if xgb_models else ""),
            "features": N_FEATURES,
            "horizons": "1d / 3d / 7d direct prediction",
            "classification_bins": ["Strong Bear (<-2%)", "Weak Bear (-2% to 0%)",
                                    "Weak Bull (0% to +2%)", "Strong Bull (>+2%)"],
        },
        "horizon_detail": {
            h: {
                "expected_return_pct": horizon_results[h]["expected_return"],
                "predicted_price": horizon_results[h]["predicted_price"],
                "direction": horizon_results[h]["direction"],
                "bin_probabilities": {
                    "strong_bear": round(horizon_results[h]["probs"][0], 4),
                    "weak_bear": round(horizon_results[h]["probs"][1], 4),
                    "weak_bull": round(horizon_results[h]["probs"][2], 4),
                    "strong_bull": round(horizon_results[h]["probs"][3], 4),
                },
            }
            for h in ["h1d", "h3d", "h7d"]
        },
    }
