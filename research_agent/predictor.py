"""
StockPulse Predictor v3 — Hybrid LSTM-Transformer Ensemble

Architecture:
  - LSTMPredictor: 2-layer LSTM + projection head (fast, good at sequences)
  - TransformerPredictor: multi-head self-attention encoder (good at patterns)
  - HybridEnsemble: weighted average of both (best of both worlds)

Features (12 total):
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

Speed optimisation:
  - LSTM: 40 epochs, hidden 64
  - Transformer: 30 epochs (converges faster)
  - Models trained in parallel-friendly order
  - Cache valid 24h per ticker
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


# ─── Hyperparameters ─────────────────────────────────────────────────────────
SEQ_LEN = 20
N_FEATURES = 12
PREDICT_DAYS = 7
CACHE_HOURS = 24
MAX_DAILY_RETURN_PCT = 5.0
HISTORY_PERIOD = "2y"

# LSTM settings
LSTM_HIDDEN = 64
LSTM_LAYERS = 2
LSTM_EPOCHS = 50
LSTM_LR = 0.005

# Transformer settings
TF_DIM = 64  # model dimension
TF_HEADS = 4  # attention heads
TF_LAYERS = 2  # encoder layers
TF_EPOCHS = 40
TF_LR = 0.003

# Ensemble: 2 LSTMs + 1 Transformer = 3 models total (fast but diverse)
N_LSTM = 2
N_TRANSFORM = 1

LOG_PATH = os.path.join(os.path.dirname(__file__), "predictions_log.json")
MODEL_CACHE_DIR = os.path.join(os.path.dirname(__file__), "model_cache")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)


# ─── Model 1: LSTM ───────────────────────────────────────────────────────────
class LSTMPredictor(nn.Module):
    """
    Standard 2-layer LSTM with LayerNorm and dropout.
    Fast to train, good at capturing sequential momentum patterns.
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
        self.fc1 = nn.Linear(LSTM_HIDDEN, LSTM_HIDDEN // 2)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(LSTM_HIDDEN // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = self.norm(out[:, -1, :])
        h = self.drop(self.relu(self.fc1(last)))
        return self.fc2(h)


# ─── Model 2: Transformer ────────────────────────────────────────────────────
class TransformerPredictor(nn.Module):
    """
    Multi-head self-attention encoder for stock prediction.

    Why Transformer works differently from LSTM:
    - LSTM processes sequences left-to-right, weighing recent steps more
    - Transformer attends to ALL positions simultaneously via self-attention
    - This means it can learn "when price crosses MA20 AND RSI > 70, what happens"
      even if those two events are 15 days apart in the window
    - LSTM would dilute the MA20 signal by the time it sees the RSI signal

    Together they capture both sequential momentum (LSTM) and
    pattern recognition across the full window (Transformer).
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

        # Output head
        self.norm = nn.LayerNorm(TF_DIM)
        self.fc1 = nn.Linear(TF_DIM, TF_DIM // 2)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.1)
        self.fc2 = nn.Linear(TF_DIM // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, SEQ_LEN, N_FEATURES)
        h = self.input_proj(x) + self.pe  # add positional encoding
        h = self.encoder(h)
        # Take the last timestep (most recent day's representation)
        out = self.norm(h[:, -1, :])
        out = self.drop(self.relu(self.fc1(out)))
        return self.fc2(out)


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
) -> np.ndarray:
    """
    12-feature matrix. Falls back gracefully if volume/OHLC not available.
    """
    n = len(prices)
    features = np.zeros((n, N_FEATURES), dtype=np.float64)

    if volumes is None:
        volumes = np.ones(n)
    if highs is None:
        highs = prices * 1.005
    if lows is None:
        lows = prices * 0.995

    # 0: normalised price
    p_min, p_max = prices.min(), prices.max()
    rng = p_max - p_min if p_max != p_min else 1.0
    features[:, 0] = (prices - p_min) / rng

    # 1: daily returns
    rets = np.zeros(n)
    rets[1:] = (prices[1:] - prices[:-1]) / prices[:-1] * 100
    features[:, 1] = (np.clip(rets, -10, 10) + 10) / 20

    # 2: 5-day z-score
    for i in range(n):
        w = prices[max(0, i - 4) : i + 1]
        mu, s = w.mean(), w.std()
        z = (prices[i] - mu) / s if s > 0 else 0.0
        features[i, 2] = np.clip(z, -3, 3) / 6 + 0.5

    # 3: sentiment
    features[:, 3] = float(np.clip((sentiment_score + 1.0) / 2.0, 0.0, 1.0))

    # 4: realised volatility (20-day)
    for i in range(n):
        w = prices[max(0, i - 19) : i + 1]
        if len(w) > 1:
            rv = float(np.std(np.diff(w) / w[:-1] * 100))
        else:
            rv = 0.0
        features[i, 4] = float(np.clip(rv / 4.0, 0.0, 1.0))

    # 5: RSI-14
    features[:, 5] = _rsi(prices)

    # 6: MACD
    features[:, 6] = _macd(prices)

    # 7: Bollinger position
    features[:, 7] = _bollinger(prices)

    # 8: SMA-20 distance
    features[:, 8] = _sma_dist(prices)

    # 9: Rate of Change
    features[:, 9] = _roc(prices)

    # 10: Volume ratio
    features[:, 10] = _vol_ratio(volumes)

    # 11: macro fear (cross-asset volatility proxy)
    for i in range(n):
        w = prices[max(0, i - 19) : i + 1]
        if len(w) > 1:
            rv = float(np.std(np.diff(w) / w[:-1] * 100))
        else:
            rv = 0.0
        features[i, 11] = float(np.clip(rv / 6.0, 0.0, 1.0))

    return features


def _build_return_targets(prices: np.ndarray) -> np.ndarray:
    r = np.zeros(len(prices))
    r[1:] = (prices[1:] - prices[:-1]) / prices[:-1] * 100
    return (np.clip(r, -10, 10) + 10) / 20


def _make_sequences(
    features: np.ndarray, targets: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    X, y = [], []
    for i in range(len(features) - SEQ_LEN):
        X.append(features[i : i + SEQ_LEN])
        y.append(targets[i + SEQ_LEN])
    return (
        torch.tensor(np.array(X), dtype=torch.float32),
        torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(-1),
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
    return {
        "closes": closes[:n],
        "volumes": volumes[:n],
        "highs": highs[:n],
        "lows": lows[:n],
        "hist": hist,
    }


# ─── Training ─────────────────────────────────────────────────────────────────
def _train_lstm(
    X: torch.Tensor, y: torch.Tensor, seed: int, lr_mult: float = 1.0
) -> LSTMPredictor:
    torch.manual_seed(seed)
    model = LSTMPredictor()
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=LSTM_LR * lr_mult, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=LSTM_EPOCHS, eta_min=1e-5
    )
    criterion = nn.HuberLoss(delta=0.5)
    model.train()
    for _ in range(LSTM_EPOCHS):
        optimiser.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        scheduler.step()
    return model


def _train_transformer(
    X: torch.Tensor, y: torch.Tensor, seed: int
) -> TransformerPredictor:
    torch.manual_seed(seed + 9999)
    model = TransformerPredictor()
    optimiser = torch.optim.AdamW(model.parameters(), lr=TF_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=TF_EPOCHS, eta_min=1e-5
    )
    criterion = nn.HuberLoss(delta=0.5)
    model.train()
    for _ in range(TF_EPOCHS):
        optimiser.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        scheduler.step()
    return model


# ─── Ensemble Rollout ─────────────────────────────────────────────────────────
def _ensemble_predict(models: list, seq_tensor: torch.Tensor) -> float:
    """Average predictions from all models (LSTM + Transformer)."""
    preds = []
    for m in models:
        m.eval()
        with torch.no_grad():
            preds.append(m(seq_tensor).item())
    return float(np.mean(preds))


def _decode_return(
    avg_norm: float,
    momentum: float,
    hist_vol: float,
    rsi_val: float = 0.5,
    macd_val: float = 0.5,
    noise_std: float = 0.0,
) -> float:
    """
    Decode normalised model output → daily % return.

    Key fix: amplify deviation from neutral (0.5) by 4x so that even
    small model signals produce visible price movement. Without this
    amplification, a model outputting 0.52 (slightly bullish) would
    produce only 0.4% move which rounds to flat on a chart.

    Uses RSI + MACD as directional confirmation signals.
    Adds calibrated noise to produce jagged natural-looking movement.
    """
    deviation = avg_norm - 0.5
    model_pct = deviation * 4.0 * 20  # 4x amplification (was 3x)

    # RSI bias: overbought → bearish nudge, oversold → bullish nudge
    rsi_bias = 0.0
    if rsi_val > 0.7:
        rsi_bias = -hist_vol * 0.4
    elif rsi_val < 0.3:
        rsi_bias = hist_vol * 0.4

    # MACD directional bias
    macd_bias = (macd_val - 0.5) * hist_vol * 0.6

    # Noise for jagged visual (calibrated to 25% of historical vol)
    noise = np.random.normal(0, noise_std) if noise_std > 0 else 0.0

    # Weighted blend
    ret_pct = (
        model_pct * 0.35
        + momentum * 0.35
        + rsi_bias * 0.15
        + macd_bias * 0.10
        + noise * 0.05
    )

    vol_cap = (
        min(MAX_DAILY_RETURN_PCT, hist_vol * 2.5)
        if hist_vol > 0
        else MAX_DAILY_RETURN_PCT
    )
    return float(np.clip(ret_pct, -vol_cap, vol_cap))


def _next_row(
    nxt_price: float, prev_row: list, ctx: list, p_min: float, rng: float, sent_n: float
) -> list:
    """Build next feature row for autoregressive step."""
    nxt_norm = (nxt_price - p_min) / rng if rng > 0 else 0.5
    prev_p = prev_row[0] * rng + p_min
    ret_pct = (nxt_price - prev_p) / prev_p * 100 if prev_p > 0 else 0.0
    ret_n = (np.clip(ret_pct, -10, 10) + 10) / 20

    recent = ctx[-4:] + [nxt_price]
    mu, sigma = np.mean(recent), np.std(recent)
    z_n = np.clip((nxt_price - mu) / sigma if sigma > 0 else 0.0, -3, 3) / 6 + 0.5

    if len(ctx) > 1:
        rv = (
            float(np.std(np.diff(ctx[-19:]) / np.array(ctx[-19:])[:-1] * 100))
            if len(ctx) >= 20
            else 0.0
        )
        vol_feat = float(np.clip(rv / 4.0, 0.0, 1.0))
        mac_fear = float(np.clip(rv / 6.0, 0.0, 1.0))
    else:
        vol_feat = prev_row[4] if len(prev_row) > 4 else 0.5
        mac_fear = prev_row[11] if len(prev_row) > 11 else 0.0

    sma_dist = (
        np.clip(
            (nxt_price - np.mean(ctx[-20:])) / np.mean(ctx[-20:]) * 100 / 10 + 0.5, 0, 1
        )
        if len(ctx) >= 20
        else 0.5
    )

    # Carry technical indicators forward (they change slowly)
    rsi_c = prev_row[5] if len(prev_row) > 5 else 0.5
    macd_c = prev_row[6] if len(prev_row) > 6 else 0.5
    boll_c = prev_row[7] if len(prev_row) > 7 else 0.5
    volr_c = prev_row[10] if len(prev_row) > 10 else 0.5
    roc_c = np.clip(ret_pct / 20 + 0.5, 0, 1)

    return [
        nxt_norm,
        ret_n,
        z_n,
        sent_n,
        vol_feat,
        rsi_c,
        macd_c,
        boll_c,
        sma_dist,
        roc_c,
        volr_c,
        mac_fear,
    ]


def _rollout(
    models: list, features: np.ndarray, prices: np.ndarray, n_days: int
) -> List[float]:
    """
    Autoregressive rollout using LSTM + Transformer ensemble.
    Majority-vote direction scaling reduces overconfident moves.
    """
    p_min, p_max = prices.min(), prices.max()
    rng = p_max - p_min if p_max != p_min else 1.0

    feat_window = features[-SEQ_LEN:].tolist()
    ctx = list(prices[-30:])
    last_price = float(prices[-1])
    future: List[float] = []

    hist_rets = np.diff(prices) / prices[:-1] * 100
    hist_vol = float(np.std(hist_rets[-30:])) if len(hist_rets) >= 30 else 1.0
    momentum = float(np.mean(hist_rets[-5:])) if len(hist_rets) >= 5 else 0.0
    sent_n = float(features[-1, 3]) if features.shape[1] > 3 else 0.5
    last_rsi = float(features[-1, 5]) if features.shape[1] > 5 else 0.5
    last_macd = float(features[-1, 6]) if features.shape[1] > 6 else 0.5
    noise_std = hist_vol * 0.25  # 25% of historical vol for jagged movement

    for m in models:
        m.eval()

    with torch.no_grad():
        for _ in range(n_days):
            seq = torch.tensor(feat_window[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0)

            # Get all predictions
            preds = [m(seq).item() for m in models]
            avg_norm = float(np.mean(preds))

            # Direction vote: dampen move when models disagree
            votes = [1 if p > 0.5 else -1 for p in preds]
            agreement = abs(sum(votes)) / len(votes)

            ret_pct = _decode_return(
                avg_norm, momentum, hist_vol, last_rsi, last_macd, noise_std
            )
            ret_pct *= 0.5 + 0.5 * agreement

            nxt_price = last_price * (1 + ret_pct / 100)
            future.append(round(float(nxt_price), 4))

            ctx.append(nxt_price)
            feat_window.append(
                _next_row(nxt_price, feat_window[-1], ctx, p_min, rng, sent_n)
            )
            last_price = nxt_price
            momentum *= 0.82

    return future


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
) -> list[str]:
    reasons = [
        f"AI model forecasts a {direction.upper()} of {pct_change:.2f}% "
        f"over the next 7 trading days (confidence: {confidence}%)."
    ]
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
            pd = pred["date"]
            if pd in existing:
                continue
            try:
                pd_date = datetime.strptime(pd, "%Y-%m-%d").date()
            except ValueError:
                continue
            if pd_date > today:
                continue
            actual = prices_by_date.get(pd)
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
                        "date": pd,
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
        return {"fine_tuned": False, "reason": f"load_error: {e}"}

    optimiser = torch.optim.Adam(model.parameters(), lr=FT_LR)
    criterion = nn.HuberLoss(delta=0.5)
    pre_losses, post_losses, used_keys = [], [], []

    for sample in new_samples:
        idx = sample["idx"]
        actual_ret_pct = (
            (sample["actual"] - sample["predicted"]) / sample["predicted"]
        ) * 100
        actual_ret_n = float((np.clip(actual_ret_pct, -10, 10) + 10) / 20)
        target = torch.tensor([[actual_ret_n]], dtype=torch.float32)
        seq = torch.tensor(
            features[idx - SEQ_LEN : idx], dtype=torch.float32
        ).unsqueeze(0)

        model.eval()
        with torch.no_grad():
            pre_losses.append(criterion(model(seq), target).item())
        model.train()
        for _ in range(FT_EPOCHS):
            optimiser.zero_grad()
            loss = criterion(model(seq), target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
        model.eval()
        with torch.no_grad():
            post_losses.append(criterion(model(seq), target).item())
        used_keys.append(sample["key"])

    try:
        torch.save(model.state_dict(), cache_path)
        with open(meta_path, "w", encoding="utf-8") as mf:
            json.dump(
                {
                    "saved_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
                    "ticker": ticker.upper(),
                    "fine_tuned": True,
                    "outcomes_used": ft_count_total + len(used_keys),
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
        hist = ohlcv["hist"]
        all_dates = [d.strftime("%Y-%m-%d") for d in hist.index][: len(all_prices)]

        if len(all_prices) < SEQ_LEN + days_back + 5:
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
        )
        train_targets = _build_return_targets(train_prices)
        X, y = _make_sequences(train_features, train_targets)

        seed = sum(ord(c) * (i + 1) for i, c in enumerate(ticker.upper())) % (2**31)
        np.random.seed(seed % (2**31))

        models = [
            _train_lstm(X, y, seed + i, [0.8, 1.2][i % 2]) for i in range(N_LSTM)
        ] + [_train_transformer(X, y, seed)]

        predicted_prices = _rollout(models, train_features, train_prices, days_back)

        last_train = float(train_prices[-1])
        direction = "rise" if predicted_prices[-1] > last_train else "fall"
        actual_dir = "rise" if float(actual_prices[-1]) > last_train else "fall"

        rows, hits, total_err = [], 0, 0.0
        for i, (pred_p, act_p, date) in enumerate(
            zip(predicted_prices, actual_prices, actual_dates)
        ):
            act_p = round(float(act_p), 4)
            error = round(((act_p - pred_p) / pred_p) * 100, 2) if pred_p != 0 else 0.0
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
                    "date": date,
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
        }

    except Exception as exc:
        return {"ticker": ticker.upper(), "error": str(exc)}


# ─── Main Predict ─────────────────────────────────────────────────────────────
def predict_stock(ticker: str) -> Dict[str, Any]:
    """
    Hybrid LSTM-Transformer ensemble prediction.
    2 LSTMs + 1 Transformer, majority-vote direction scaling.
    12 features including RSI, MACD, Bollinger, volume, sentiment, macro fear.
    No model fit line — only future forecast returned.
    """
    try:
        ohlcv = _fetch_ohlcv(ticker)
    except Exception as exc:
        return {"ticker": ticker.upper(), "error": str(exc)}

    prices = ohlcv["closes"]
    volumes = ohlcv["volumes"]
    highs = ohlcv["highs"]
    lows = ohlcv["lows"]

    sentiment_score = get_sentiment(ticker.upper()) or 0.0
    sentiment_used = sentiment_score != 0.0

    features = _build_feature_matrix(
        prices, sentiment_score=sentiment_score, volumes=volumes, highs=highs, lows=lows
    )
    targets = _build_return_targets(prices)
    X, y = _make_sequences(features, targets)

    seed = sum(ord(c) * (i + 1) for i, c in enumerate(ticker.upper())) % (2**31)
    np.random.seed(seed % (2**31))

    try:
        check_past_predictions(ticker.upper())
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

    # Cache: save/load first LSTM only (Transformer trains fast anyway)
    cache_path = os.path.join(MODEL_CACHE_DIR, f"{ticker.upper()}.pt")
    meta_path = os.path.join(MODEL_CACHE_DIR, f"{ticker.upper()}.json")
    cache_valid = False
    cached_lstm = LSTMPredictor()

    if os.path.exists(cache_path) and os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as mf:
                meta = json.load(mf)
            saved_at = datetime.strptime(meta["saved_at"], "%Y-%m-%d %H:%M UTC")
            if (datetime.utcnow() - saved_at).total_seconds() / 3600 < CACHE_HOURS:
                cached_lstm.load_state_dict(
                    torch.load(cache_path, map_location="cpu", weights_only=True)
                )
                cache_valid = True
        except Exception:
            cache_valid = False

    # Build ensemble: 2 LSTMs + 1 Transformer
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
                    },
                    mf,
                )
        except Exception:
            pass

    transformer = _train_transformer(X, y, seed)
    models = [lstm1, lstm2, transformer]

    # Rollout future predictions
    future_prices = _rollout(models, features, prices, PREDICT_DAYS)

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
    last_hist = hist_prices[-1]
    day1_pct = (future_prices[0] - last_hist) / last_hist * 100
    overall_pct = (future_prices[-1] - last_hist) / last_hist * 100
    weighted = day1_pct * 0.6 + overall_pct * 0.4
    direction = "rise" if weighted >= 0 else "fall"
    pct_change = abs(overall_pct)
    confidence = round(50 + 45 * (1 - math.exp(-abs(weighted) / 1.5)), 1)

    signals = _compute_signals(prices, hist_prices, hist_dates)
    reasoning = _build_reasoning(
        direction, pct_change, confidence, signals, sentiment_score
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
            "architecture": "Hybrid LSTM-Transformer Ensemble",
            "models": f"{N_LSTM} LSTM + {N_TRANSFORM} Transformer",
            "features": N_FEATURES,
        },
    }
