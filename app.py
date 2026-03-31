#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Breadth Quant Engine Ultimate v10 High Contrast Holistic Gate
- Historical gates + range map + oscillator-aware repair matrix
- Holistic TSI (customizable) with clear high-contrast gauge
- Unified Holistic Gate = Sweet-Spot Gates + Holistic TSI
- Backtest vs RSP / SPY using holistic TSI signals
"""

from __future__ import annotations

import io
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# -----------------------------
# App config / style
# -----------------------------
st.set_page_config(page_title="Breadth Quant Engine Ultimate v10 High Contrast Holistic Gate", layout="wide", page_icon="📈")

CUSTOM_CSS = """
<style>
:root{
  --bg:#050814;--panel:#0d1428;--panel2:#111a33;--text:#ffffff;--muted:#c7d2fe;
  --green:#22c55e;--yellow:#f59e0b;--red:#ef4444;--blue:#38bdf8;--orange:#fb923c;
}
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {background:var(--bg); color:var(--text);}
.block-container{padding-top:1rem;padding-bottom:2rem;max-width:1560px;}
.main-title{padding:1.1rem 1.25rem;border-radius:18px;background:linear-gradient(135deg, rgba(56,189,248,.22), rgba(167,139,250,.24));border:1px solid rgba(255,255,255,.18);margin-bottom:1rem;box-shadow:0 12px 34px rgba(0,0,0,.28);}
.soft-card{background:linear-gradient(180deg, rgba(14,20,40,.99), rgba(8,13,27,.99));border:1px solid rgba(255,255,255,.18);border-radius:18px;padding:1rem 1rem .95rem 1rem;box-shadow:0 14px 36px rgba(0,0,0,.30);margin-bottom:1rem;}
.score-title{color:#dbeafe;font-size:1.02rem;font-weight:900;letter-spacing:.02em;text-transform:uppercase;}
.score-value{font-size:2.55rem;line-height:1.05;font-weight:1000;color:#ffffff;margin:.35rem 0;text-shadow:0 1px 0 rgba(0,0,0,.4);}
.score-value-sm{font-size:1.55rem;line-height:1.15;font-weight:950;color:#ffffff;margin:.2rem 0;}
.small-muted{color:#cbd5e1;font-size:.92rem;font-weight:600;}
.tiny-muted{color:#dbe4ff;font-size:.84rem;font-weight:600;}
.pill{display:inline-block;padding:.34rem .68rem;border-radius:999px;font-size:.84rem;font-weight:900;border:1px solid rgba(255,255,255,.12);margin-right:.4rem;margin-bottom:.32rem;color:#fff;}
.pill-green{background:#166534;color:#dcfce7;}
.pill-yellow{background:#92400e;color:#fef3c7;}
.pill-red{background:#991b1b;color:#fee2e2;}
.pill-blue{background:#1d4ed8;color:#dbeafe;}
.pill-orange{background:#9a3412;color:#ffedd5;}
.setup-line{padding:.55rem .65rem;border-radius:12px;margin:.35rem 0;border:1px solid rgba(255,255,255,.10);font-weight:700;color:#fff;}
.setup-pass{background:rgba(34,197,94,.18);}
.setup-near{background:rgba(245,158,11,.20);}
.setup-far{background:rgba(239,68,68,.18);}
.bar{height:10px;border-radius:999px;background:rgba(255,255,255,.08);overflow:hidden;margin-top:.35rem;}
.fill-green{height:100%;background:linear-gradient(90deg,#16a34a,#4ade80);}
.fill-yellow{height:100%;background:linear-gradient(90deg,#d97706,#fbbf24);}
.fill-red{height:100%;background:linear-gradient(90deg,#dc2626,#f87171);}
.gauge-wrap{padding:.3rem 0 .15rem 0;}
.gauge-track{height:24px;border-radius:999px;background:linear-gradient(90deg,#7f1d1d 0%, #dc2626 16%, #d97706 34%, #64748b 50%, #d97706 66%, #16a34a 84%, #14532d 100%);position:relative;overflow:hidden;border:1px solid rgba(255,255,255,.14);}
.gauge-marker{position:absolute;top:-2px;width:11px;height:28px;border-radius:8px;background:#fff;box-shadow:0 0 0 2px rgba(255,255,255,.16),0 2px 12px rgba(255,255,255,.35);}
.kpi-box{padding:.85rem;border-radius:16px;border:1px solid rgba(255,255,255,.14);background:rgba(255,255,255,.04);}
.state-green{background:linear-gradient(180deg, rgba(15,47,30,.98), rgba(8,20,15,.98));}
.state-yellow{background:linear-gradient(180deg, rgba(69,45,11,.98), rgba(28,20,7,.98));}
.state-red{background:linear-gradient(180deg, rgba(69,14,14,.98), rgba(26,9,9,.98));}
.state-neutral{background:linear-gradient(180deg, rgba(30,41,59,.98), rgba(13,18,30,.98));}
div[data-testid="stMetric"]{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.12);padding:.75rem .9rem;border-radius:14px;}
div[data-testid="stMetricLabel"], div[data-testid="stMetricValue"], div[data-testid="stMetricDelta"]{color:#fff !important;}
[data-testid="stDataFrame"] div[role="grid"]{font-size:15px;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown("""
<div class='main-title'>
  <div style='font-size:1.8rem;font-weight:900;'>📈 Breadth Quant Engine Ultimate v10 High Contrast Holistic Gate</div>
  <div class='small-muted'>Historical gates + range map + oscillator-aware repair matrix + holistic TSI + backtest.</div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Paths / constants
# -----------------------------
APP_DIR = Path("breadth_quant_store_v9")
APP_DIR.mkdir(exist_ok=True)
HIST_DAILY_PATH = APP_DIR / "daily_history.parquet"
HIST_WEEKLY_PATH = APP_DIR / "weekly_history.parquet"
MODEL_PATH = APP_DIR / "learned_model.json"

KEY_FEATURES = [
    "$BPSPX", "$BPSPX_%B", "$BPNYA", "$OEXA200R", "$SPXA50R", "$NYMO", "$NYSI",
    "$CPCE", "$NYHL", "$NYAD", "$SPXADP", "$TRIN", "$VIX", "RSP:SPY"
]
INVERSE_INDICATORS = {"$TRIN", "$VIX", "$CPCE", "VXX", "SPXS:SVOL"}
STATE_ORDER = ["washout", "bounce", "repair", "regime", "overheating"]
RANGE_FEATURES = ["$BPSPX", "$BPSPX_%B", "$BPNYA", "$OEXA200R", "$SPXA50R", "$NYMO", "$NYSI", "$CPCE", "$NYHL"]
OSC_FEATURES = ["$BPSPX", "$SPXA50R", "$NYMO", "$NYSI", "$BPNYA", "$OEXA200R", "$NYHL"]
TREND_WINDOWS = [1, 2, 3, 5, 10]
OUTCOME_DEFS = {
    "bounce": {"horizon": 10, "ret": 0.03, "dd": -0.03, "type": "max"},
    "repair": {"horizon": 20, "ret": 0.04, "dd": -0.05, "type": "end"},
    "regime": {"horizon": 60, "ret": 0.08, "dd": -0.08, "type": "end"},
    "fall": {"horizon": 10, "ret": -0.03, "dd": 0.03, "type": "min_end"},
}
CANARY_WEIGHTS = {
    "SPXS:SVOL": 0.24, "HYG:IEF": 0.20, "SMH:SPY": 0.18,
    "XLF:SPY": 0.12, "RSP:SPY": 0.12, "IWM:SPY": 0.10, "$VIX": 0.04
}
SYMBOL_MAP = {
    "rsp": "RSP", "ursp": "URSP", "spy": "SPY", "vxx": "VXX",
    "_bpspx": "$BPSPX", "bpspx": "$BPSPX", "_bpnya": "$BPNYA", "bpnya": "$BPNYA",
    "_oexa50r": "$OEXA50R", "oexa50r": "$OEXA50R", "_oexa150r": "$OEXA150R", "oexa150r": "$OEXA150R",
    "_oexa200r": "$OEXA200R", "oexa200r": "$OEXA200R", "_spxa50r": "$SPXA50R", "spxa50r": "$SPXA50R",
    "_nymo": "$NYMO", "nymo": "$NYMO", "_nysi": "$NYSI", "nysi": "$NYSI",
    "_cpce": "$CPCE", "cpce": "$CPCE", "_nyhl": "$NYHL", "nyhl": "$NYHL",
    "_nyad": "$NYAD", "nyad": "$NYAD", "_spxadp": "$SPXADP", "spxadp": "$SPXADP",
    "_trin": "$TRIN", "trin": "$TRIN", "_vix": "$VIX", "vix": "$VIX",
    "hyg_ief": "HYG:IEF", "hyg_tlt": "HYG:TLT", "rsp_spy": "RSP:SPY", "smh_spy": "SMH:SPY",
    "iwm_spy": "IWM:SPY", "xlf_spy": "XLF:SPY", "spxs_svol": "SPXS:SVOL",
}

TSI_BUCKETS = {
    "breadth": [
        "$BPSPX", "$BPNYA", "$SPXA50R", "$OEXA50R", "$OEXA150R", "$OEXA200R",
        "$NYMO", "$NYSI", "$NYHL", "$NYAD", "$SPXADP"
    ],
    "leadership": ["RSP", "SPY", "RSP:SPY", "SMH:SPY", "IWM:SPY", "XLF:SPY", "HYG:IEF", "HYG:TLT"],
    "risk": ["$VIX", "VXX", "$TRIN", "$CPCE", "SPXS:SVOL"],
}
TSI_BUCKET_WEIGHTS = {"breadth": 0.50, "leadership": 0.30, "risk": 0.20}
TSI_COMPONENT_WEIGHTS = {
    "$BPSPX": 1.25, "$BPNYA": 1.00, "$SPXA50R": 1.25, "$OEXA50R": 0.75, "$OEXA150R": 0.65, "$OEXA200R": 0.70,
    "$NYMO": 1.20, "$NYSI": 1.00, "$NYHL": 0.80, "$NYAD": 0.85, "$SPXADP": 0.85,
    "RSP": 0.90, "SPY": 0.70, "RSP:SPY": 1.00, "SMH:SPY": 0.85, "IWM:SPY": 0.75, "XLF:SPY": 0.65,
    "HYG:IEF": 0.90, "HYG:TLT": 0.70,
    "$VIX": 1.00, "VXX": 0.90, "$TRIN": 0.80, "$CPCE": 0.80, "SPXS:SVOL": 1.00
}

# -----------------------------
# Utils
# -----------------------------
def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return np.nan


def fmt_num(v: Any, d: int = 2) -> str:
    if pd.isna(v):
        return "n/a"
    return f"{float(v):.{d}f}"


def save_json(path: Path, data: Any) -> bool:
    try:
        path.write_text(json.dumps(data, indent=2, default=str))
        return True
    except Exception:
        return False


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


# -----------------------------
# Indicators
# -----------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1 / period, adjust=False).mean()
    ma_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def percent_b(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    denom = (upper - lower).replace(0, np.nan)
    return (series - lower) / denom


def macd_hist(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    line = ema(series, fast) - ema(series, slow)
    sig = ema(line, signal)
    return line - sig


def stoch_from_close(close: pd.Series, length: int = 14, smoothk: int = 3) -> pd.Series:
    close = pd.to_numeric(close, errors="coerce")
    lo = close.rolling(length).min()
    hi = close.rolling(length).max()
    denom = (hi - lo).replace(0, np.nan)
    k = 100 * (close - lo) / denom
    return k.rolling(smoothk).mean()


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    high = pd.to_numeric(high, errors="coerce")
    low = pd.to_numeric(low, errors="coerce")
    close = pd.to_numeric(close, errors="coerce")
    tp = (high + low + close) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))


def true_strength_index(series: pd.Series, long_len: int = 25, short_len: int = 13, signal_len: int = 7) -> Tuple[pd.Series, pd.Series]:
    series = pd.to_numeric(series, errors="coerce")
    m = series.diff()
    abs_m = m.abs()
    double_smoothed_m = ema(ema(m, long_len), short_len)
    double_smoothed_abs = ema(ema(abs_m, long_len), short_len)
    tsi = 100 * (double_smoothed_m / double_smoothed_abs.replace(0, np.nan))
    signal = ema(tsi, signal_len)
    return tsi, signal


def add_indicator_features(hist: pd.DataFrame) -> pd.DataFrame:
    out = []
    for sym, g in hist.groupby("symbol", sort=False):
        g = g.sort_values("date").copy()
        c = g["close"]
        h = g["high"] if "high" in g.columns else c
        l = g["low"] if "low" in g.columns else c
        g["pct_b20"] = percent_b(c, 20, 2.0)
        g["rsi14"] = rsi(c, 14)
        g["cci20"] = cci(h, l, c, 20)
        g["macd_hist"] = macd_hist(c)
        g["stoch14"] = stoch_from_close(c, 14, 3)
        for w in TREND_WINDOWS:
            g[f"d{w}"] = c.diff(w)
            g[f"roc{w}"] = 100 * (c / c.shift(w) - 1)
        out.append(g)
    return pd.concat(out, ignore_index=True)


# -----------------------------
# Parsing
# -----------------------------
def parse_stockcharts_csv(content: bytes) -> pd.DataFrame:
    text = content.decode("utf-8", errors="ignore")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    rows = []
    for ln in lines:
        if ln.lower().startswith(("date,", "symbol,", "ticker,")):
            continue
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 6:
            continue
        dt = pd.to_datetime(parts[0], errors="coerce")
        if pd.isna(dt):
            continue
        nums = [safe_float(x) for x in parts[1:6]]
        if all(pd.isna(x) for x in nums[:4]):
            continue
        rows.append(
            {
                "date": dt,
                "open": nums[0],
                "high": nums[1],
                "low": nums[2],
                "close": nums[3],
                "volume": nums[4],
            }
        )
    if not rows:
        raise ValueError("No rows parsed from StockCharts CSV")
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def symbol_from_filename(name: str) -> Tuple[str, str]:
    stem = Path(name).stem.strip().lower()
    timeframe = "weekly" if stem.endswith("w") or stem.endswith("_w") else "daily"
    if timeframe == "weekly":
        if stem.endswith("_w"):
            stem = stem[:-2]
        elif stem.endswith("w"):
            stem = stem[:-1]
    stem = stem.strip("_")
    sym = SYMBOL_MAP.get(stem, stem.upper())
    return sym, timeframe


def parse_stockcharts_zip(file_bytes: bytes) -> Tuple[pd.DataFrame, pd.DataFrame]:
    daily, weekly = [], []
    with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
        for name in zf.namelist():
            if name.endswith("/") or not name.lower().endswith(".csv"):
                continue
            try:
                content = zf.read(name)
                df = parse_stockcharts_csv(content)
                sym, tf = symbol_from_filename(name)
                df["symbol"] = sym
                (weekly if tf == "weekly" else daily).append(df)
            except Exception:
                continue
    if not daily:
        raise ValueError("No daily CSV parsed from ZIP")
    daily_df = pd.concat(daily, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    weekly_df = (
        pd.concat(weekly, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
        if weekly
        else pd.DataFrame(columns=daily_df.columns)
    )
    return daily_df, weekly_df


def parse_snapshot_csv(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    if "Symbol" not in df.columns:
        raise ValueError("Snapshot must include Symbol column")
    close_col = next((c for c in ["Close", "Last", "Price", "Current", "Value"] if c in df.columns), None)
    if close_col is None:
        raise ValueError("Snapshot must include close-like column")
    out = pd.DataFrame({"symbol": df["Symbol"].astype(str).str.strip(), "close": pd.to_numeric(df[close_col], errors="coerce")})
    return out.dropna(subset=["close"])


# -----------------------------
# Model helpers
# -----------------------------
def compute_future_metrics(price: pd.Series, horizon: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    vals = price.to_numpy(dtype=float)
    n = len(vals)
    idx = price.index
    if n < horizon + 1:
        nan = pd.Series(np.nan, index=idx)
        return nan, nan, nan
    x = np.arange(n)
    win_idx = x[:, None] + np.arange(1, horizon + 1)
    mask = win_idx >= n
    safe_idx = np.clip(win_idx, 0, n - 1)
    win = vals[safe_idx].astype(float)
    win[mask] = np.nan
    rets = win / vals[:, None] - 1
    valid_rows = ~np.all(np.isnan(rets), axis=1)
    end_ret = np.full(n, np.nan)
    max_gain = np.full(n, np.nan)
    max_dd = np.full(n, np.nan)
    end_ret[valid_rows] = rets[valid_rows, -1]
    if valid_rows.any():
        max_gain[valid_rows] = np.nanmax(rets[valid_rows], axis=1)
        max_dd[valid_rows] = np.nanmin(rets[valid_rows], axis=1)
    return pd.Series(end_ret, index=idx), pd.Series(max_gain, index=idx), pd.Series(max_dd, index=idx)


def build_outcomes(rsp_price: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=rsp_price.index)
    for name, cfg in OUTCOME_DEFS.items():
        end_ret, max_gain, max_dd = compute_future_metrics(rsp_price, cfg["horizon"])
        if cfg["type"] == "max":
            success = (max_gain >= cfg["ret"]) & (max_dd >= cfg["dd"])
        elif cfg["type"] == "min_end":
            success = (end_ret <= cfg["ret"]) & (max_dd <= cfg["dd"])
        else:
            success = (end_ret >= cfg["ret"]) & (max_dd >= cfg["dd"])
        out[f"{name}_success"] = success.astype(float)
    return out


def ratio_indicator_score(close: pd.Series) -> pd.Series:
    close = pd.to_numeric(close, errors="coerce")
    if close.dropna().shape[0] < 220:
        return pd.Series(dtype=float)
    mh = macd_hist(close)
    tsi_proxy = close.diff().ewm(span=20).mean() / close.diff().abs().ewm(span=20).mean() * 100
    stoch = stoch_from_close(close, 14, 3)
    cci100 = ((close - close.rolling(100).mean()) / (0.015 * (close - close.rolling(100).mean()).abs().rolling(100).mean().replace(0, np.nan)))
    df = pd.concat([mh.rename("macdh"), tsi_proxy.rename("tsi"), stoch.rename("stoch"), cci100.rename("cci")], axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    bull = (df["macdh"] > 0) & (df["tsi"] > 0) & (df["stoch"] > 50) & (df["cci"] > 0)
    bear = (df["macdh"] < 0) & (df["tsi"] < 0) & (df["stoch"] < 50) & (df["cci"] < 0)
    s = pd.Series(0.0, index=df.index)
    s[bull] = 1.0
    s[bear] = -1.0
    return s


def build_canary_from_history(daily_feat: pd.DataFrame) -> pd.DataFrame:
    piv = daily_feat.pivot(index="date", columns="symbol", values="close")
    score_map = {}
    for sym, wt in CANARY_WEIGHTS.items():
        if sym not in piv.columns:
            continue
        s = piv[sym].dropna()
        sc = ratio_indicator_score(s)
        if sc.empty:
            continue
        if sym in INVERSE_INDICATORS:
            sc = -sc
        score_map[sym] = sc
    if not score_map:
        return pd.DataFrame()
    common = None
    for s in score_map.values():
        common = s.index if common is None else common.intersection(s.index)
    if common is None or len(common) == 0:
        return pd.DataFrame()
    df = pd.DataFrame({k: v.loc[common] for k, v in score_map.items()}).dropna(how="all")
    weights = pd.Series({k: CANARY_WEIGHTS[k] for k in df.columns})
    weights = weights / weights.sum()
    comp = (df * weights).sum(axis=1)
    sign = np.sign(comp.replace(0, np.nan))
    align = np.sign(df).replace(0, np.nan).eq(sign, axis=0).mean(axis=1).fillna(0)
    strength = df.abs().mean(axis=1).fillna(0)
    conf = ((0.6 * align) + (0.4 * strength)) * 100
    return pd.DataFrame({"canary_comp": comp, "canary_conf": conf})


def direction_hints(feature: str, state: str) -> List[str]:
    if state == "fall":
        if feature in INVERSE_INDICATORS:
            return ["gte", "lte"]
        return ["lte", "gte"]
    if feature in INVERSE_INDICATORS:
        return ["lte", "gte"] if state in {"repair", "regime"} else ["gte", "lte"]
    return ["gte", "lte"] if state in {"repair", "regime"} else ["lte", "gte"]


def learn_single_gates(df: pd.DataFrame, features: List[str], label: str, min_support: int = 30) -> List[dict]:
    y = df[f"{label}_success"].astype(float)
    base = float(y.mean())
    gates = []
    for feat in features:
        s = pd.to_numeric(df[feat], errors="coerce")
        valid = s.notna() & y.notna()
        sv, yv = s[valid], y[valid]
        if len(sv) < max(80, min_support * 2):
            continue
        qs = sorted(set(float(x) for x in sv.quantile(np.linspace(0.15, 0.85, 15)).dropna()))
        for direction in direction_hints(feat, label):
            for thr in qs:
                mask = sv >= thr if direction == "gte" else sv <= thr
                support = int(mask.sum())
                if support < min_support:
                    continue
                hit = float(yv[mask].mean())
                if hit <= base:
                    continue
                score = (hit - base) * math.sqrt(support)
                gates.append({"feature": feat, "direction": direction, "threshold": thr, "support": support, "hit_rate": hit, "base_rate": base, "lift": hit / base if base > 0 else np.nan, "score": score})
    gates = sorted(gates, key=lambda x: (x["score"], x["lift"], x["support"]), reverse=True)
    top, used = [], set()
    for g in gates:
        if g["feature"] in used:
            continue
        used.add(g["feature"])
        top.append(g)
        if len(top) >= 12:
            break
    return top


def learn_combo_gates(df: pd.DataFrame, label: str, singles: List[dict], min_support: int = 25) -> List[dict]:
    y = df[f"{label}_success"].astype(float)
    base = float(y.mean())
    combos = []
    for i in range(len(singles)):
        for j in range(i + 1, min(len(singles), i + 6)):
            g1, g2 = singles[i], singles[j]
            if g1["feature"] == g2["feature"]:
                continue
            s1 = pd.to_numeric(df[g1["feature"]], errors="coerce")
            s2 = pd.to_numeric(df[g2["feature"]], errors="coerce")
            m1 = s1 >= g1["threshold"] if g1["direction"] == "gte" else s1 <= g1["threshold"]
            m2 = s2 >= g2["threshold"] if g2["direction"] == "gte" else s2 <= g2["threshold"]
            mask = m1 & m2 & y.notna()
            support = int(mask.sum())
            if support < min_support:
                continue
            hit = float(y[mask].mean())
            if hit <= base:
                continue
            score = (hit - base) * math.sqrt(support)
            combos.append({"gates": [g1, g2], "support": support, "hit_rate": hit, "base_rate": base, "lift": hit / base if base > 0 else np.nan, "score": score})
    return sorted(combos, key=lambda x: (x["score"], x["lift"], x["support"]), reverse=True)[:6]


def gate_pass(cur: float, gate: dict) -> bool:
    if pd.isna(cur):
        return False
    return cur >= gate["threshold"] if gate["direction"] == "gte" else cur <= gate["threshold"]


def summarize_bands(df: pd.DataFrame, label: str) -> Dict[str, Dict[str, float]]:
    hit = df[df[f"{label}_success"] == 1.0]
    res = {}
    for feat in KEY_FEATURES:
        if feat not in hit.columns:
            continue
        s = pd.to_numeric(hit[feat], errors="coerce").dropna()
        if len(s) < 10:
            continue
        res[feat] = {"median": float(s.median()), "q25": float(s.quantile(0.25)), "q75": float(s.quantile(0.75)), "count": int(len(s))}
    return res


def band_distance_score(x: float, q25: float, med: float, q75: float) -> float:
    if pd.isna(x) or pd.isna(q25) or pd.isna(med) or pd.isna(q75):
        return np.nan
    iqr = max(abs(q75 - q25), 1e-6)
    if q25 <= x <= q75:
        d = abs(x - med) / iqr
        return max(0.72, 1.0 - 0.28 * d)
    d = min(abs(x - med) / iqr, 3.0)
    return max(0.0, 0.72 - 0.24 * (d - 1.0))


def score_bands(snapshot: Dict[str, float], bands: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows, totals = [], {}
    for label in ["bounce", "repair", "regime"]:
        vals = []
        for feat, meta in bands.get(label, {}).items():
            cur = safe_float(snapshot.get(feat, np.nan))
            q25, med, q75 = meta["q25"], meta["median"], meta["q75"]
            if feat in INVERSE_INDICATORS and pd.notna(cur):
                cur = -cur
                q25, med, q75 = -q75, -med, -q25
            sc = band_distance_score(cur, q25, med, q75)
            rows.append({"Outcome": label.title(), "Feature": feat, "Current": cur, "Median": med, "Q25": q25, "Q75": q75, "BandScore": sc})
            if pd.notna(sc):
                vals.append(sc)
        totals[label] = 100 * np.mean(vals) if vals else np.nan
    return pd.DataFrame(rows), totals


@dataclass
class ClusterArtifacts:
    scaler_mean: List[float]
    scaler_scale: List[float]
    features: List[str]
    centroids: List[List[float]]
    cluster_names: Dict[str, str]
    silhouette_score: float = 0.0


def assign_cluster_names(stats_df: pd.DataFrame) -> Dict[int, str]:
    names = {}
    for idx, row in stats_df.iterrows():
        bounce = row.get("bounce_rate", 0)
        repair = row.get("repair_rate", 0)
        regime = row.get("regime_rate", 0)
        fall = row.get("fall_rate", 0)
        bpspx_pct_b = row.get("$BPSPX_%B_median", np.nan)
        if fall >= max(repair, bounce, regime) and fall > 0.30:
            names[idx] = "Deterioration cluster"
        elif regime >= max(repair, bounce, fall) and regime > 0.30:
            names[idx] = "Durable regime"
        elif repair >= max(regime, bounce, fall) and repair > 0.22:
            names[idx] = "Repair cluster"
        elif bounce >= max(regime, repair, fall) and bounce > 0.35:
            names[idx] = "Bounce cluster"
        elif pd.notna(bpspx_pct_b) and bpspx_pct_b < 0.15:
            names[idx] = "Capitulation / washout"
        else:
            names[idx] = "Mixed / transitional"
    return names


def build_clusters(base: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, ClusterArtifacts]:
    feat_df = base[features].apply(pd.to_numeric, errors="coerce").dropna()
    n_clusters = max(3, min(6, max(3, len(feat_df) // 40)))
    scaler = StandardScaler()
    X = scaler.fit_transform(feat_df)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20, max_iter=300)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else 0.0
    rows = []
    for cl in sorted(np.unique(labels)):
        sub_idx = feat_df.index[labels == cl]
        rows.append(
            {
                "cluster": int(cl),
                "samples": int((labels == cl).sum()),
                "bounce_rate": float(base.loc[sub_idx, "bounce_success"].mean()),
                "repair_rate": float(base.loc[sub_idx, "repair_success"].mean()),
                "regime_rate": float(base.loc[sub_idx, "regime_success"].mean()),
                "fall_rate": float(base.loc[sub_idx, "fall_success"].mean()),
                **{f"{f}_median": float(base.loc[sub_idx, f].median()) for f in features if f in base.columns},
            }
        )
    stats_df = pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)
    names = assign_cluster_names(stats_df)
    stats_df["cluster_name"] = stats_df["cluster"].map(names)
    art = ClusterArtifacts(
        scaler_mean=scaler.mean_.tolist(),
        scaler_scale=scaler.scale_.tolist(),
        features=features,
        centroids=km.cluster_centers_.tolist(),
        cluster_names={str(k): v for k, v in names.items()},
        silhouette_score=float(sil),
    )
    return stats_df, art


def predict_cluster(current: Dict[str, float], artifacts: ClusterArtifacts) -> Tuple[Optional[int], Optional[str], Optional[float]]:
    vals = []
    for feat in artifacts.features:
        v = safe_float(current.get(feat, np.nan))
        if pd.isna(v):
            return None, None, None
        vals.append(v)
    x = np.array(vals)
    scaled = (x - np.array(artifacts.scaler_mean)) / np.where(np.array(artifacts.scaler_scale) == 0, 1, np.array(artifacts.scaler_scale))
    cents = np.array(artifacts.centroids)
    dists = np.sqrt(((cents - scaled) ** 2).sum(axis=1))
    cl = int(np.argmin(dists))
    conf = 1.0 / (1.0 + float(dists[cl]))
    return cl, artifacts.cluster_names.get(str(cl), f"Cluster {cl}"), conf


def build_model_from_history(daily: pd.DataFrame, weekly: pd.DataFrame) -> Dict[str, Any]:
    daily_feat = add_indicator_features(daily)
    weekly_feat = add_indicator_features(weekly) if not weekly.empty else weekly.copy()
    piv_close = daily_feat.pivot(index="date", columns="symbol", values="close")
    if "RSP" not in piv_close.columns:
        raise ValueError("RSP daily history is required")
    base = piv_close.copy()
    piv_pb = daily_feat.pivot(index="date", columns="symbol", values="pct_b20")
    piv_rsi = daily_feat.pivot(index="date", columns="symbol", values="rsi14")
    piv_cci = daily_feat.pivot(index="date", columns="symbol", values="cci20")
    piv_mh = daily_feat.pivot(index="date", columns="symbol", values="macd_hist")

    for sym in piv_pb.columns:
        base[f"{sym}_%B"] = piv_pb[sym]
    for sym in piv_rsi.columns:
        base[f"{sym}_RSI14"] = piv_rsi[sym]
    for sym in piv_cci.columns:
        base[f"{sym}_CCI20"] = piv_cci[sym]
    for sym in piv_mh.columns:
        base[f"{sym}_MACDH"] = piv_mh[sym]

    outcomes = build_outcomes(piv_close["RSP"].dropna())
    base = base.join(outcomes, how="inner").dropna(subset=["RSP"])

    features = [c for c in base.columns if c not in [f"{k}_success" for k in OUTCOME_DEFS]]
    features = [c for c in features if base[c].notna().sum() >= 80]

    learned = {"states": {}, "bands": {}, "meta": {"rows": int(len(base))}}
    for state in ["bounce", "repair", "regime", "fall"]:
        singles = learn_single_gates(base, features, state)
        combos = learn_combo_gates(base, state, singles)
        learned["states"][state] = {"singles": singles, "combos": combos, "base_rate": float(base[f"{state}_success"].mean())}
        if state != "fall":
            learned["bands"][state] = summarize_bands(base, state)

    cluster_features = [f for f in KEY_FEATURES if f in base.columns]
    cluster_stats, cluster_art = build_clusters(base.dropna(subset=cluster_features).copy(), cluster_features)
    canary_hist = build_canary_from_history(daily_feat)

    learned["clusters"] = {
        "features": cluster_art.features,
        "mean": cluster_art.scaler_mean,
        "scale": cluster_art.scaler_scale,
        "centroids": cluster_art.centroids,
        "names": cluster_art.cluster_names,
        "silhouette": cluster_art.silhouette_score,
        "stats": cluster_stats.to_dict(orient="records"),
    }
    learned["canary_hist"] = canary_hist.reset_index().rename(columns={"index": "date"}).to_dict(orient="records") if not canary_hist.empty else []

    daily_feat.to_parquet(HIST_DAILY_PATH, index=False)
    if not weekly_feat.empty:
        weekly_feat.to_parquet(HIST_WEEKLY_PATH, index=False)
    save_json(MODEL_PATH, learned)
    return learned


# -----------------------------
# Current snapshot / evaluation
# -----------------------------
def build_snapshot_from_history_and_csv(daily_feat: pd.DataFrame, snapshot_df: Optional[pd.DataFrame]) -> Tuple[Dict[str, float], Dict[str, float], pd.Timestamp]:
    piv_close = daily_feat.pivot(index="date", columns="symbol", values="close")
    piv_pb = daily_feat.pivot(index="date", columns="symbol", values="pct_b20")
    piv_rsi = daily_feat.pivot(index="date", columns="symbol", values="rsi14")
    piv_cci = daily_feat.pivot(index="date", columns="symbol", values="cci20")
    piv_mh = daily_feat.pivot(index="date", columns="symbol", values="macd_hist")
    latest_date = pd.to_datetime(daily_feat["date"]).max()
    prior_date = pd.to_datetime(daily_feat[daily_feat["date"] < latest_date]["date"]).max()

    if snapshot_df is not None and not snapshot_df.empty:
        snap_map = dict(zip(snapshot_df["symbol"], snapshot_df["close"]))
        hist_recalc = daily_feat.copy()
        last_mask = hist_recalc["date"] == latest_date
        for sym, val in snap_map.items():
            m = last_mask & (hist_recalc["symbol"] == sym)
            hist_recalc.loc[m, "close"] = val
            hist_recalc.loc[m, "high"] = np.maximum(hist_recalc.loc[m, "high"], val)
            hist_recalc.loc[m, "low"] = np.minimum(hist_recalc.loc[m, "low"], val)
        recalc = add_indicator_features(hist_recalc)
        piv_pb = recalc.pivot(index="date", columns="symbol", values="pct_b20")
        piv_rsi = recalc.pivot(index="date", columns="symbol", values="rsi14")
        piv_cci = recalc.pivot(index="date", columns="symbol", values="cci20")
        piv_mh = recalc.pivot(index="date", columns="symbol", values="macd_hist")
        piv_close = recalc.pivot(index="date", columns="symbol", values="close")

    snapshot, prior = {}, {}
    for sym in piv_close.columns:
        cur = safe_float(piv_close.loc[latest_date, sym]) if latest_date in piv_close.index else np.nan
        prv = safe_float(piv_close.loc[prior_date, sym]) if pd.notna(prior_date) and prior_date in piv_close.index else np.nan
        snapshot[sym] = cur
        prior[sym] = prv
        if sym in piv_pb.columns:
            snapshot[f"{sym}_%B"] = safe_float(piv_pb.loc[latest_date, sym])
            prior[f"{sym}_%B"] = safe_float(piv_pb.loc[prior_date, sym]) if pd.notna(prior_date) else np.nan
        if sym in piv_rsi.columns:
            snapshot[f"{sym}_RSI14"] = safe_float(piv_rsi.loc[latest_date, sym])
            prior[f"{sym}_RSI14"] = safe_float(piv_rsi.loc[prior_date, sym]) if pd.notna(prior_date) else np.nan
        if sym in piv_cci.columns:
            snapshot[f"{sym}_CCI20"] = safe_float(piv_cci.loc[latest_date, sym])
            prior[f"{sym}_CCI20"] = safe_float(piv_cci.loc[prior_date, sym]) if pd.notna(prior_date) else np.nan
        if sym in piv_mh.columns:
            snapshot[f"{sym}_MACDH"] = safe_float(piv_mh.loc[latest_date, sym])
            prior[f"{sym}_MACDH"] = safe_float(piv_mh.loc[prior_date, sym]) if pd.notna(prior_date) else np.nan
    return snapshot, prior, latest_date


def proxy_nymo(snapshot: Dict[str, float], prev_snapshot: Dict[str, float]) -> Dict[str, float]:
    nyad = safe_float(snapshot.get("$NYAD", np.nan))
    spxadp = safe_float(snapshot.get("$SPXADP", np.nan))
    prev_nyad = safe_float(prev_snapshot.get("$NYAD", np.nan))
    prev_spxadp = safe_float(prev_snapshot.get("$SPXADP", np.nan))
    cur_raw = 0.6 * (0 if pd.isna(nyad) else nyad) + 0.4 * (0 if pd.isna(spxadp) else spxadp)
    prev_raw = 0.6 * (0 if pd.isna(prev_nyad) else prev_nyad) + 0.4 * (0 if pd.isna(prev_spxadp) else prev_spxadp)
    cur = 100 * np.tanh(cur_raw / 1600.0)
    prev = 100 * np.tanh(prev_raw / 1600.0)
    state = "Deep washout" if cur <= -70 else "Negative but repairing" if cur <= -20 else "Neutral / crossing" if cur <= 20 else "Positive thrust"
    return {"value": float(cur), "delta": float(cur - prev), "state": state}


def evaluate_state(snapshot: Dict[str, float], state_model: Dict[str, Any]) -> Dict[str, Any]:
    singles = []
    for g in state_model.get("singles", []):
        cur = safe_float(snapshot.get(g["feature"], np.nan))
        singles.append({**g, "current": cur, "passed": gate_pass(cur, g)})
    combos = []
    for combo in state_model.get("combos", []):
        passes = [gate_pass(safe_float(snapshot.get(g["feature"], np.nan)), g) for g in combo["gates"]]
        combos.append({**combo, "passed": all(passes)})
    pass_frac = float(np.mean([x["passed"] for x in singles])) if singles else 0.0
    base = state_model.get("base_rate", np.nan)
    passed_hits = [x["hit_rate"] for x in singles if x["passed"]]
    passed_combo_hits = [x["hit_rate"] for x in combos if x["passed"]]
    prob = base
    if passed_hits:
        prob = 0.55 * np.mean(passed_hits) + 0.25 * (np.mean(passed_combo_hits) if passed_combo_hits else base) + 0.20 * base
        prob = float(np.clip(prob * (0.65 + 0.35 * pass_frac), 0, 1)) if pd.notna(prob) else np.nan
    return {"prob": prob, "pass_frac": pass_frac, "base_rate": base, "passed_singles": [x for x in singles if x["passed"]], "passed_combos": [x for x in combos if x["passed"]], "all_singles": singles}


def compute_repair_oscillator_matrix(snapshot: Dict[str, float], prev_snapshot: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for sym in OSC_FEATURES:
        level = safe_float(snapshot.get(sym, np.nan))
        bb = safe_float(snapshot.get(f"{sym}_%B", np.nan))
        rsi_v = safe_float(snapshot.get(f"{sym}_RSI14", np.nan))
        cci_v = safe_float(snapshot.get(f"{sym}_CCI20", np.nan))
        mh = safe_float(snapshot.get(f"{sym}_MACDH", np.nan))
        prev_bb = safe_float(prev_snapshot.get(f"{sym}_%B", np.nan))
        prev_rsi = safe_float(prev_snapshot.get(f"{sym}_RSI14", np.nan))
        prev_cci = safe_float(prev_snapshot.get(f"{sym}_CCI20", np.nan))
        prev_mh = safe_float(prev_snapshot.get(f"{sym}_MACDH", np.nan))
        bb_up = pd.notna(bb) and pd.notna(prev_bb) and bb > prev_bb
        rsi_up = pd.notna(rsi_v) and pd.notna(prev_rsi) and rsi_v > prev_rsi
        cci_up = pd.notna(cci_v) and pd.notna(prev_cci) and cci_v > prev_cci
        mh_up = pd.notna(mh) and pd.notna(prev_mh) and mh > prev_mh
        score = int(bb_up) + int(rsi_up) + int(cci_up) + int(mh_up)
        if score >= 3:
            state = "Repairing strongly"
        elif score == 2:
            state = "Repairing"
        elif score == 1:
            state = "Mixed"
        else:
            state = "Not repairing"
        rows.append(
            {
                "Feature": sym,
                "Level": level,
                "BB%": bb,
                "BB% Δ": None if pd.isna(bb) or pd.isna(prev_bb) else bb - prev_bb,
                "RSI14": rsi_v,
                "RSI Δ": None if pd.isna(rsi_v) or pd.isna(prev_rsi) else rsi_v - prev_rsi,
                "CCI20": cci_v,
                "CCI Δ": None if pd.isna(cci_v) or pd.isna(prev_cci) else cci_v - prev_cci,
                "MACDH": mh,
                "MACDH Δ": None if pd.isna(mh) or pd.isna(prev_mh) else mh - prev_mh,
                "Repair Score": score,
                "Oscillator State": state,
            }
        )
    return pd.DataFrame(rows)


def build_range_map(snapshot: Dict[str, float], bands: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for feat in RANGE_FEATURES:
        cur = safe_float(snapshot.get(feat, np.nan))
        b = bands.get("bounce", {}).get(feat)
        r = bands.get("repair", {}).get(feat)
        g = bands.get("regime", {}).get(feat)
        if not any([b, r, g]):
            continue

        def rng(meta):
            return (meta.get("q25"), meta.get("q75"), meta.get("median")) if meta else (np.nan, np.nan, np.nan)

        b25, b75, bc = rng(b)
        r25, r75, rc = rng(r)
        g25, g75, gc = rng(g)
        state = "Transitional"
        if pd.notna(cur):
            if pd.notna(g75) and cur > g75:
                state = "Overheating"
            elif pd.notna(g25) and g25 <= cur <= g75:
                state = "Regime"
            elif pd.notna(r25) and r25 <= cur <= r75:
                state = "Repair"
            elif pd.notna(b25) and b25 <= cur <= b75:
                state = "Bounce"
            else:
                if feat in INVERSE_INDICATORS:
                    state = "Washout / Fall Risk" if (pd.notna(b25) and cur > b75) else "Transitional"
                else:
                    state = "Washout / Fall Risk" if (pd.notna(b25) and cur < b25) else "Transitional"
        rows.append(
            {
                "Feature": feat,
                "Current": cur,
                "Bounce Range": f"{fmt_num(b25,3)} – {fmt_num(b75,3)}",
                "Bounce Center": bc,
                "Repair Range": f"{fmt_num(r25,3)} – {fmt_num(r75,3)}",
                "Repair Center": rc,
                "Regime Range": f"{fmt_num(g25,3)} – {fmt_num(g75,3)}",
                "Regime Center": gc,
                "State Ladder": state,
            }
        )
    return pd.DataFrame(rows)


def nearest_confirmation_from_ranges(snapshot: Dict[str, float], bands: Dict[str, Any], bullish: bool = True) -> List[dict]:
    setups = []
    if bullish:
        for feat in ["$BPSPX_%B", "$SPXA50R", "$BPSPX", "$BPNYA", "$NYMO", "$CPCE", "$TRIN", "RSP:SPY"]:
            cur = safe_float(snapshot.get(feat, np.nan))
            if pd.isna(cur):
                continue
            if feat in bands.get("repair", {}):
                q25 = bands["repair"][feat]["q25"]
                q75 = bands["repair"][feat]["q75"]
                if feat in INVERSE_INDICATORS:
                    if cur > q75:
                        gap = cur - q75
                        setups.append({"feature": feat, "op": "≤", "threshold": q75, "current": cur, "gap": gap, "state": "repair"})
                else:
                    if cur < q25:
                        setups.append({"feature": feat, "op": "≥", "threshold": q25, "current": cur, "gap": q25 - cur, "state": "repair"})
        return sorted(setups, key=lambda x: x["gap"])[:6]
    else:
        for feat in ["$OEXA200R", "$SPXA50R", "$BPSPX", "$NYMO", "$NYHL", "$TRIN", "$CPCE", "$VIX", "RSP:SPY"]:
            cur = safe_float(snapshot.get(feat, np.nan))
            if pd.isna(cur):
                continue
            if feat in bands.get("bounce", {}):
                q25 = bands["bounce"][feat]["q25"]
                if feat in INVERSE_INDICATORS:
                    if cur < q25:
                        setups.append({"feature": feat, "op": "≥", "threshold": q25, "current": cur, "gap": q25 - cur, "state": "fall"})
                else:
                    if cur > q25:
                        setups.append({"feature": feat, "op": "≤", "threshold": q25, "current": cur, "gap": cur - q25, "state": "fall"})
        return sorted(setups, key=lambda x: x["gap"])[:6]


def setup_progress(cur: float, threshold: float, op: str) -> Tuple[str, float]:
    if pd.isna(cur) or pd.isna(threshold):
        return "far", 0.0
    if op == "≥":
        if cur >= threshold:
            return "pass", 1.0
        ratio = max(0.0, cur / threshold) if threshold != 0 else 0.0
    else:
        if cur <= threshold:
            return "pass", 1.0
        ratio = max(0.0, threshold / cur) if cur != 0 else 0.0
    if ratio >= 0.95:
        return "near", ratio
    return "far", ratio


def render_setup_lines(setups: List[dict], title: str):
    st.markdown(f"**{title}**")
    if not setups:
        st.write("No clean setup triggers available.")
        return
    for s in setups:
        state, prog = setup_progress(s["current"], s["threshold"], s["op"])
        css = "setup-pass" if state == "pass" else "setup-near" if state == "near" else "setup-far"
        fill = "fill-green" if state == "pass" else "fill-yellow" if state == "near" else "fill-red"
        text = f"{s['feature']} {s['op']} {fmt_num(s['threshold'],3)} (now {fmt_num(s['current'],3)}) • state: {s['state']}"
        st.markdown(f"<div class='setup-line {css}'>{text}<div class='bar'><div class='{fill}' style='width:{max(3, int(prog*100))}%;'></div></div></div>", unsafe_allow_html=True)


def classify_signal(state_scores: Dict[str, Any], canary: Dict[str, Any], recovery_score: float, cluster_name: Optional[str], osc_df: pd.DataFrame) -> Dict[str, Any]:
    bounce = state_scores["bounce"]
    repair = state_scores["repair"]
    regime = state_scores["regime"]
    fall = state_scores["fall"]
    repairing_count = int((osc_df["Repair Score"] >= 2).sum()) if not osc_df.empty else 0
    long_flag = (
        ((bounce["pass_frac"] >= 0.55 and bounce["prob"] >= max(0.40, bounce["base_rate"] + 0.08))
         or (repair["pass_frac"] >= 0.50 and repair["prob"] >= max(0.30, repair["base_rate"] + 0.06)))
        and recovery_score >= 45
        and repairing_count >= 3
        and canary["label"] != "Risk-Off"
    ) or ((regime["pass_frac"] >= 0.50 and regime["prob"] >= max(0.30, regime["base_rate"] + 0.05) and canary["label"] != "Risk-Off"))
    short_flag = (fall["pass_frac"] >= 0.50 and fall["prob"] >= max(0.25, fall["base_rate"] + 0.05) and canary["label"] != "Risk-On" and recovery_score < 40 and repairing_count <= 2)
    signal = "LONG" if long_flag else "SHORT" if short_flag else "HOLD"
    reasons = []
    if signal == "LONG":
        reasons.append(f"Bounce/Repair probability favorable ({max(bounce['prob'], repair['prob']):.0%})")
        reasons.append(f"Repair breadth: {repairing_count}/{len(osc_df)} oscillators repairing")
    elif signal == "SHORT":
        reasons.append(f"Fall probability dominant ({fall['prob']:.0%})")
        reasons.append(f"Repair breadth weak: {repairing_count}/{len(osc_df)}")
    else:
        reasons.append("Mixed signals; no strong edge")
        reasons.append(f"Repair breadth: {repairing_count}/{len(osc_df)} oscillators repairing")
    reasons.append(f"Canary: {canary['label']}")
    reasons.append(f"Recovery: {recovery_score:.0f}")
    if cluster_name:
        reasons.append(f"Cluster: {cluster_name}")
    return {"signal": signal, "reasons": reasons, "bounce_prob": bounce["prob"], "repair_prob": repair["prob"], "regime_prob": regime["prob"], "fall_prob": fall["prob"]}


# -----------------------------
# Holistic TSI
# -----------------------------
def orient_series_for_risk(series: pd.Series, symbol: str) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    return -series if symbol in INVERSE_INDICATORS else series


def compute_bucket_composite(piv_close: pd.DataFrame, symbols: List[str], long_len: int, short_len: int, signal_len: int) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    component_rows = []
    aligned_tsi = {}
    aligned_sig = {}

    for sym in symbols:
        if sym not in piv_close.columns:
            continue
        raw = piv_close[sym].dropna()
        if raw.shape[0] < max(long_len + short_len + signal_len + 10, 60):
            continue
        oriented = orient_series_for_risk(raw, sym)
        tsi, sig = true_strength_index(oriented, long_len, short_len, signal_len)
        if tsi.dropna().empty or sig.dropna().empty:
            continue
        common = tsi.index.intersection(sig.index)
        tsi = tsi.loc[common]
        sig = sig.loc[common]
        w = TSI_COMPONENT_WEIGHTS.get(sym, 1.0)
        aligned_tsi[sym] = tsi * w
        aligned_sig[sym] = sig * w

        gap = tsi - sig
        slope3 = tsi.diff(3)
        component_rows.append(
            pd.DataFrame(
                {
                    "date": common,
                    "symbol": sym,
                    "weight": w,
                    "tsi": tsi.values,
                    "signal": sig.values,
                    "gap": gap.values,
                    "slope3": slope3.values,
                    "above_signal": (tsi > sig).astype(int).values,
                    "above_zero": (tsi > 0).astype(int).values,
                }
            )
        )

    if not aligned_tsi:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.DataFrame()

    tsi_df = pd.DataFrame(aligned_tsi).sort_index()
    sig_df = pd.DataFrame(aligned_sig).sort_index()
    weights = pd.Series({c: TSI_COMPONENT_WEIGHTS.get(c, 1.0) for c in tsi_df.columns})
    weights = weights / weights.sum()
    bucket_tsi = tsi_df.mul(weights, axis=1).sum(axis=1)
    bucket_sig = sig_df.mul(weights, axis=1).sum(axis=1)
    comp_df = pd.concat(component_rows, ignore_index=True)
    return bucket_tsi, bucket_sig, comp_df


def compute_holistic_tsi_history(daily_feat: pd.DataFrame, long_len: int, short_len: int, signal_len: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    piv_close = daily_feat.pivot(index="date", columns="symbol", values="close").sort_index()

    bucket_out = {}
    component_tables = []

    for bucket_name, syms in TSI_BUCKETS.items():
        b_tsi, b_sig, comp_df = compute_bucket_composite(piv_close, syms, long_len, short_len, signal_len)
        if not b_tsi.empty:
            bucket_out[f"{bucket_name}_tsi"] = b_tsi
            bucket_out[f"{bucket_name}_signal"] = b_sig
        if not comp_df.empty:
            comp_df["bucket"] = bucket_name
            component_tables.append(comp_df)

    if not bucket_out:
        return pd.DataFrame(), pd.DataFrame()

    hist = pd.DataFrame(bucket_out).sort_index()
    used_bucket_weights = {k: v for k, v in TSI_BUCKET_WEIGHTS.items() if f"{k}_tsi" in hist.columns}
    total_w = sum(used_bucket_weights.values())
    used_bucket_weights = {k: v / total_w for k, v in used_bucket_weights.items()}

    hist["holistic_tsi"] = 0.0
    hist["holistic_signal"] = 0.0
    for k, w in used_bucket_weights.items():
        hist["holistic_tsi"] += hist[f"{k}_tsi"] * w
        hist["holistic_signal"] += hist[f"{k}_signal"] * w

    # participation metrics
    if component_tables:
        comp_all = pd.concat(component_tables, ignore_index=True)
        part = (
            comp_all.groupby("date")
            .agg(
                components=("symbol", "nunique"),
                above_signal=("above_signal", "sum"),
                above_zero=("above_zero", "sum"),
                mean_gap=("gap", "mean"),
                mean_slope3=("slope3", "mean"),
            )
            .sort_index()
        )
        hist = hist.join(part, how="left")
        hist["pct_above_signal"] = 100 * hist["above_signal"] / hist["components"]
        hist["pct_above_zero"] = 100 * hist["above_zero"] / hist["components"]

        bucket_part = (
            comp_all.groupby(["date", "bucket"])
            .agg(
                comps=("symbol", "nunique"),
                above_sig=("above_signal", "sum"),
                above_zero=("above_zero", "sum"),
            )
            .reset_index()
        )
        for bucket in sorted(bucket_part["bucket"].unique()):
            sub = bucket_part[bucket_part["bucket"] == bucket].set_index("date")
            hist[f"{bucket}_pct_above_signal"] = 100 * sub["above_sig"] / sub["comps"]
            hist[f"{bucket}_pct_above_zero"] = 100 * sub["above_zero"] / sub["comps"]

    hist["gap"] = hist["holistic_tsi"] - hist["holistic_signal"]
    hist["gap_prev"] = hist["gap"].shift(1)
    hist["slope1"] = hist["holistic_tsi"].diff(1)
    hist["slope3"] = hist["holistic_tsi"].diff(3)
    hist["cross_up"] = (hist["gap"] > 0) & (hist["gap_prev"] <= 0)
    hist["cross_down"] = (hist["gap"] < 0) & (hist["gap_prev"] >= 0)
    hist["zero_up"] = (hist["holistic_tsi"] > 0) & (hist["holistic_tsi"].shift(1) <= 0)
    hist["zero_down"] = (hist["holistic_tsi"] < 0) & (hist["holistic_tsi"].shift(1) >= 0)
    hist["distance_to_cross_pct"] = 100 * (1 - (hist["gap"].abs() / (hist["holistic_tsi"].abs() + hist["holistic_signal"].abs() + 1e-6)).clip(0, 1))
    comp_df_final = pd.concat(component_tables, ignore_index=True) if component_tables else pd.DataFrame()
    return hist, comp_df_final


def holistic_gauge_state(row: pd.Series) -> Dict[str, Any]:
    if row is None or row.empty or pd.isna(row.get("holistic_tsi", np.nan)) or pd.isna(row.get("holistic_signal", np.nan)):
        return {"label": "No data", "action": "n/a", "emoji": "⚪", "pct": 0.0, "trade_bias": "n/a"}

    tsi = safe_float(row["holistic_tsi"])
    sig = safe_float(row["holistic_signal"])
    gap = tsi - sig
    slope = safe_float(row.get("slope3", np.nan))
    pct_align = safe_float(row.get("pct_above_signal", np.nan))
    proximity = float(np.clip(row.get("distance_to_cross_pct", 0.0), 0, 100))

    just_crossed_up = bool(row.get("cross_up", False))
    just_crossed_down = bool(row.get("cross_down", False))
    above_zero = tsi > 0
    below_zero = tsi < 0

    # confidence / completion style percentage
    base_pct = min(100.0, max(0.0, 0.55 * proximity + 0.45 * (0 if pd.isna(pct_align) else pct_align)))

    if just_crossed_up and below_zero:
        return {"label": "Bounce Triggered", "action": "Probe long", "emoji": "🟡", "pct": max(88.0, base_pct), "trade_bias": "early_bull"}
    if gap > 0 and below_zero and slope > 0:
        return {"label": "Repair Underway", "action": "Add only on strength", "emoji": "🟡", "pct": max(75.0, base_pct), "trade_bias": "repair"}
    if just_crossed_up and above_zero:
        return {"label": "Regime Up Confirmed", "action": "Full bull bias", "emoji": "🟢", "pct": 100.0, "trade_bias": "regime_up"}
    if gap > 0 and above_zero and slope >= 0:
        return {"label": "Bullish Regime", "action": "Hold / add winners", "emoji": "🟢", "pct": max(80.0, base_pct), "trade_bias": "bull"}
    if gap > 0 and above_zero and slope < 0:
        return {"label": "Overheating", "action": "Trim into strength", "emoji": "🟠", "pct": max(70.0, 100 - proximity / 2), "trade_bias": "overheat"}
    if just_crossed_down and above_zero:
        return {"label": "Fall Triggered", "action": "Trim / hedge", "emoji": "🔴", "pct": 100.0, "trade_bias": "fall"}
    if gap < 0 and above_zero:
        return {"label": "Near Bearish Cross", "action": "Reduce risk", "emoji": "🟠", "pct": max(85.0, base_pct), "trade_bias": "weakening"}
    if just_crossed_down and below_zero:
        return {"label": "Regime Down Confirmed", "action": "Defensive / short bias", "emoji": "🔴", "pct": 100.0, "trade_bias": "regime_down"}
    if gap < 0 and below_zero:
        return {"label": "Bearish Regime", "action": "Stay defensive", "emoji": "🔴", "pct": max(80.0, base_pct), "trade_bias": "bear"}
    if abs(gap) <= 1.0 and slope > 0:
        return {"label": "Near Bullish Cross", "action": "Probe if other signals agree", "emoji": "🟡", "pct": max(90.0, base_pct), "trade_bias": "near_bull"}
    if abs(gap) <= 1.0 and slope < 0:
        return {"label": "Near Bearish Cross", "action": "Tighten stops", "emoji": "🟠", "pct": max(90.0, base_pct), "trade_bias": "near_bear"}
    return {"label": "Neutral / Mixed", "action": "Hold / wait", "emoji": "⚪", "pct": base_pct, "trade_bias": "neutral"}


def compute_lead_lag_events(hist: pd.DataFrame, benchmark_price: pd.Series, bench_tsi: pd.Series, look_forward: int = 10, bounce_thr: float = 0.03, fall_thr: float = -0.03) -> pd.DataFrame:
    df = hist.copy()
    bench = pd.DataFrame({"price": benchmark_price, "bench_tsi": bench_tsi}).dropna()
    df = df.join(bench, how="inner")
    if df.empty or len(df) < look_forward + 20:
        return pd.DataFrame()

    out_rows = []
    for dt in df.index[:-look_forward]:
        future_window = df.loc[dt:].iloc[: look_forward + 1]
        if len(future_window) < look_forward + 1:
            continue
        fwd_ret = future_window["price"].iloc[-1] / future_window["price"].iloc[0] - 1
        bench_gap = future_window["bench_tsi"].iloc[1:] - future_window["bench_tsi"].iloc[1:].shift(1)

        if fwd_ret >= bounce_thr:
            prior = df.loc[:dt].tail(20)
            h_cross = prior[prior["cross_up"]]
            h_zero = prior[prior["zero_up"]]
            b_cross_dates = prior.index[(prior["bench_tsi"] > prior["bench_tsi"].shift(1)) & (prior["bench_tsi"].shift(1) <= 0)]
            out_rows.append(
                {
                    "event": "Bounce / Rally",
                    "event_date": dt,
                    "holistic_cross_lead_days": (dt - h_cross.index[-1]).days if not h_cross.empty else np.nan,
                    "holistic_zero_lead_days": (dt - h_zero.index[-1]).days if not h_zero.empty else np.nan,
                    "benchmark_tsi_turn_lead_days": (dt - b_cross_dates[-1]).days if len(b_cross_dates) else np.nan,
                    "forward_return": fwd_ret,
                }
            )
        elif fwd_ret <= fall_thr:
            prior = df.loc[:dt].tail(20)
            h_cross = prior[prior["cross_down"]]
            h_zero = prior[prior["zero_down"]]
            b_cross_dates = prior.index[(prior["bench_tsi"] < prior["bench_tsi"].shift(1)) & (prior["bench_tsi"].shift(1) >= 0)]
            out_rows.append(
                {
                    "event": "Fall / Breakdown",
                    "event_date": dt,
                    "holistic_cross_lead_days": (dt - h_cross.index[-1]).days if not h_cross.empty else np.nan,
                    "holistic_zero_lead_days": (dt - h_zero.index[-1]).days if not h_zero.empty else np.nan,
                    "benchmark_tsi_turn_lead_days": (dt - b_cross_dates[-1]).days if len(b_cross_dates) else np.nan,
                    "forward_return": fwd_ret,
                }
            )
    return pd.DataFrame(out_rows)




def compute_holistic_gate(state_scores: Dict[str, Any], gauge: Dict[str, Any]) -> Dict[str, Any]:
    bounce_p = float(state_scores.get("bounce", {}).get("prob") or 0.0)
    repair_p = float(state_scores.get("repair", {}).get("prob") or 0.0)
    regime_p = float(state_scores.get("regime", {}).get("prob") or 0.0)
    fall_p = float(state_scores.get("fall", {}).get("prob") or 0.0)
    label = gauge.get("label", "Neutral / Mixed")
    bias = gauge.get("trade_bias", "neutral")

    state = "Neutral"
    action = "Hold / wait"
    reasons: List[str] = []

    if bias in {"regime_down", "bear"} and fall_p >= max(bounce_p, repair_p, regime_p):
        state, action = "Regime Down", "Defensive / short bias"
        reasons = [f"Fall gate dominant ({fall_p:.0%})", f"Holistic TSI bearish ({label})"]
    elif bias in {"fall", "weakening", "near_bear"} and fall_p >= 0.22:
        state, action = "Fall", "Reduce risk / hedge"
        reasons = [f"Fall gate rising ({fall_p:.0%})", f"Momentum rolling over ({label})"]
    elif bias in {"regime_up", "bull"} and regime_p >= max(bounce_p, repair_p, fall_p):
        state, action = "Regime Up", "Full bull bias"
        reasons = [f"Regime gate strong ({regime_p:.0%})", f"Holistic TSI bullish ({label})"]
    elif bias == "overheat" and regime_p >= 0.20:
        state, action = "Overheating", "Trim winners"
        reasons = [f"Regime still favorable ({regime_p:.0%})", f"Momentum flattening ({label})"]
    elif bias in {"repair", "early_bull", "near_bull"} and repair_p >= max(fall_p, 0.18):
        if bias in {"early_bull", "near_bull"}:
            state, action = "Bounce", "Probe long"
        else:
            state, action = "Repair", "Add only on strength"
        reasons = [f"Repair gate favorable ({repair_p:.0%})", f"Momentum transition in progress ({label})"]
    elif bounce_p >= max(fall_p, 0.18) and repair_p >= 0.15:
        state, action = "Bounce", "Probe long"
        reasons = [f"Bounce/repair gates supportive ({max(bounce_p, repair_p):.0%})", f"Momentum not fully confirmed yet ({label})"]
    elif fall_p > regime_p and fall_p > repair_p:
        state, action = "Fall", "Reduce risk / hedge"
        reasons = [f"Fall gate strongest ({fall_p:.0%})", f"Momentum mixed ({label})"]
    else:
        reasons = ["Mixed sweet-spot probabilities", f"Holistic TSI: {label}"]

    conf = max(bounce_p, repair_p, regime_p, fall_p) * 100
    conf = float(np.clip(0.65 * conf + 0.35 * float(gauge.get("pct", 0.0)), 0, 100))
    emoji_map = {
        "Bounce": "🟡", "Repair": "🟡", "Regime Up": "🟢",
        "Overheating": "🟠", "Fall": "🔴", "Regime Down": "🔴", "Neutral": "⚪"
    }
    return {"state": state, "action": action, "confidence": conf, "emoji": emoji_map.get(state, "⚪"), "reasons": reasons}


def render_holistic_gate_card(gate: Dict[str, Any], state_scores: Dict[str, Any]):
    state = gate.get("state", "Neutral")
    css = "state-green" if state == "Regime Up" else "state-red" if state in {"Fall", "Regime Down"} else "state-yellow" if state in {"Bounce", "Repair", "Overheating"} else "state-neutral"
    st.markdown(f"<div class='soft-card {css}'>", unsafe_allow_html=True)
    st.markdown("<div class='score-title'>Holistic Gate</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='score-value-sm'>{gate['emoji']} {gate['state']}</div>", unsafe_allow_html=True)
    st.markdown(f"<span class='pill pill-blue'>Confidence: {gate['confidence']:.0f}%</span>", unsafe_allow_html=True)
    st.markdown(f"<span class='pill pill-yellow'>Trade: {gate['action']}</span>", unsafe_allow_html=True)
    for k in ["bounce", "repair", "regime", "fall"]:
        prob = float(state_scores.get(k, {}).get("prob") or 0.0) * 100
        pill = "pill-red" if k == "fall" else "pill-green" if k in {"repair", "regime"} else "pill-blue"
        st.markdown(f"<span class='pill {pill}'>{k.title()}: {prob:.1f}%</span>", unsafe_allow_html=True)
    if gate.get("reasons"):
        st.markdown(f"<div class='tiny-muted'>{' | '.join(gate['reasons'][:3])}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def run_holistic_backtest(
    hist: pd.DataFrame,
    daily_feat: pd.DataFrame,
    benchmark_symbol: str = "RSP",
    mode: str = "Long / Cash",
    signal_logic: str = "Cross vs signal",
    require_zero_filter: bool = False,
    min_align: float = 50.0,
    lookback_years: Optional[int] = 1,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    piv = daily_feat.pivot(index="date", columns="symbol", values="close").sort_index()
    if benchmark_symbol not in piv.columns:
        raise ValueError(f"{benchmark_symbol} not found in daily history")

    bench = piv[benchmark_symbol].rename("price").dropna().to_frame()
    bench["ret"] = bench["price"].pct_change()
    bt = hist.join(bench, how="inner").dropna(subset=["holistic_tsi", "holistic_signal", "price"])
    if lookback_years is not None and not bt.empty:
        cutoff = bt.index.max() - pd.DateOffset(years=int(lookback_years))
        bt = bt.loc[bt.index >= cutoff]
    if bt.empty:
        return pd.DataFrame(), {}

    if benchmark_symbol in bt.columns:
        pass

    gap = bt["holistic_tsi"] - bt["holistic_signal"]

    if signal_logic == "Cross vs signal":
        long_entry = (gap > 0) & (gap.shift(1) <= 0)
        exit_signal = (gap < 0) & (gap.shift(1) >= 0)
    elif signal_logic == "Cross zero":
        long_entry = (bt["holistic_tsi"] > 0) & (bt["holistic_tsi"].shift(1) <= 0)
        exit_signal = (bt["holistic_tsi"] < 0) & (bt["holistic_tsi"].shift(1) >= 0)
    else:
        long_entry = ((gap > 0) & (gap.shift(1) <= 0)) | ((bt["holistic_tsi"] > 0) & (bt["holistic_tsi"].shift(1) <= 0))
        exit_signal = ((gap < 0) & (gap.shift(1) >= 0)) | ((bt["holistic_tsi"] < 0) & (bt["holistic_tsi"].shift(1) >= 0))

    short_entry = (gap < 0) & (gap.shift(1) >= 0)
    short_exit = (gap > 0) & (gap.shift(1) <= 0)

    if require_zero_filter:
        long_entry = long_entry & (bt["holistic_tsi"] > 0)
        short_entry = short_entry & (bt["holistic_tsi"] < 0)

    if "pct_above_signal" in bt.columns:
        long_entry = long_entry & (bt["pct_above_signal"] >= min_align)
        if mode == "Long / Short":
            short_entry = short_entry & ((100 - bt["pct_above_signal"]) >= min_align)

    position = []
    pos = 0
    for idx, row in bt.iterrows():
        if mode == "Long / Cash":
            if long_entry.loc[idx]:
                pos = 1
            elif exit_signal.loc[idx]:
                pos = 0
        else:
            if long_entry.loc[idx]:
                pos = 1
            elif short_entry.loc[idx]:
                pos = -1
            elif pos == 1 and exit_signal.loc[idx]:
                pos = 0
            elif pos == -1 and short_exit.loc[idx]:
                pos = 0
        position.append(pos)

    bt["position"] = pd.Series(position, index=bt.index).shift(1).fillna(0)
    bt["strategy_ret"] = bt["position"] * bt["ret"].fillna(0)
    bt["buy_hold_ret"] = bt["ret"].fillna(0)
    bt["strategy_equity"] = (1 + bt["strategy_ret"]).cumprod()
    bt["buy_hold_equity"] = (1 + bt["buy_hold_ret"]).cumprod()
    bt["price_tsi"], bt["price_tsi_signal"] = true_strength_index(bt["price"], 25, 13, 7)

    total_ret = bt["strategy_equity"].iloc[-1] - 1
    bh_ret = bt["buy_hold_equity"].iloc[-1] - 1
    cagr = bt["strategy_equity"].iloc[-1] ** (252 / max(len(bt), 1)) - 1
    bh_cagr = bt["buy_hold_equity"].iloc[-1] ** (252 / max(len(bt), 1)) - 1
    max_dd = (bt["strategy_equity"] / bt["strategy_equity"].cummax() - 1).min()
    bh_max_dd = (bt["buy_hold_equity"] / bt["buy_hold_equity"].cummax() - 1).min()
    trades = int(((bt["position"] != bt["position"].shift(1)) & (bt["position"] != 0)).sum())
    win_days = float((bt["strategy_ret"] > 0).mean() * 100)

    lead_events = compute_lead_lag_events(bt[["holistic_tsi", "holistic_signal", "cross_up", "cross_down", "zero_up", "zero_down"]], bt["price"], bt["price_tsi"])
    lead_summary = {}
    if not lead_events.empty:
        for evt in ["Bounce / Rally", "Fall / Breakdown"]:
            sub = lead_events[lead_events["event"] == evt]
            if not sub.empty:
                lead_summary[evt] = {
                    "holistic_cross_lead_days": float(sub["holistic_cross_lead_days"].mean()),
                    "holistic_zero_lead_days": float(sub["holistic_zero_lead_days"].mean()),
                    "benchmark_tsi_turn_lead_days": float(sub["benchmark_tsi_turn_lead_days"].mean()),
                    "count": int(len(sub)),
                }

    stats = {
        "strategy_return": total_ret,
        "buy_hold_return": bh_ret,
        "strategy_cagr": cagr,
        "buy_hold_cagr": bh_cagr,
        "strategy_max_dd": max_dd,
        "buy_hold_max_dd": bh_max_dd,
        "trades": trades,
        "win_day_pct": win_days,
        "lead_summary": lead_summary,
    }
    return bt, stats


# -----------------------------
# Rendering helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def build_model_from_history_bytes(file_bytes: bytes) -> Dict[str, Any]:
    daily, weekly = parse_stockcharts_zip(file_bytes)
    return build_model_from_history(daily, weekly)


def load_model() -> Optional[Dict[str, Any]]:
    return load_json(MODEL_PATH, None)


def render_score_card(title: str, value: float, pct: Optional[float] = None):
    pill_class = "pill-blue"
    if title.lower() == "bounce":
        pill_class = "pill-blue"
    elif title.lower() in {"repair", "regime"}:
        pill_class = "pill-green"
    elif title.lower() == "fall":
        pill_class = "pill-red"
    st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='score-title'>{title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='score-value'>{int(round(value)) if pd.notna(value) else 'n/a'}</div>", unsafe_allow_html=True)
    if pct is not None and pd.notna(pct):
        st.markdown(f"<span class='pill {pill_class}'>{pct:.1f}%</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_signal_box(signal: str, text: str):
    klass = "pill-green" if signal == "LONG" else "pill-red" if signal == "SHORT" else "pill-yellow"
    st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='score-title'>Daily Verdict</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='score-value-sm'>{signal}</div>", unsafe_allow_html=True)
    st.markdown(f"<span class='pill {klass}'>{text}</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def gauge_color_class(label: str) -> str:
    if any(x in label for x in ["Regime Up", "Bullish Regime"]):
        return "state-green"
    if any(x in label for x in ["Bounce", "Repair", "Near Bullish"]):
        return "state-yellow"
    if any(x in label for x in ["Fall", "Bearish", "Regime Down"]):
        return "state-red"
    if "Overheating" in label:
        return "state-yellow"
    return ""


def render_holistic_gauge(gauge: Dict[str, Any], row: pd.Series):
    pct = float(np.clip(gauge["pct"], 0, 100))
    left_pos = max(1, min(98, pct))
    css = gauge_color_class(gauge["label"]) or "state-neutral"
    st.markdown(f"<div class='soft-card {css}'>", unsafe_allow_html=True)
    st.markdown("<div class='score-title'>Holistic TSI Gauge</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='score-value-sm'>{gauge['emoji']} {gauge['label']} — {pct:.0f}%</div>", unsafe_allow_html=True)
    st.markdown(f"<span class='pill pill-blue'>Action: {gauge['action']}</span>", unsafe_allow_html=True)
    st.markdown(f"<span class='pill pill-yellow'>TSI: {fmt_num(row.get('holistic_tsi', np.nan),2)}</span>", unsafe_allow_html=True)
    st.markdown(f"<span class='pill pill-blue'>Signal: {fmt_num(row.get('holistic_signal', np.nan),2)}</span>", unsafe_allow_html=True)
    st.markdown(f"<span class='pill pill-green'>Above signal: {fmt_num(row.get('pct_above_signal', np.nan),1)}%</span>", unsafe_allow_html=True)
    st.markdown(f"<span class='pill pill-orange'>Slope(3): {fmt_num(row.get('slope3', np.nan),2)}</span>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='gauge-wrap'><div class='gauge-track'><div class='gauge-marker' style='left: calc({left_pos}% - 5px);'></div></div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='tiny-muted'>Below zero bullish cross = Bounce / Repair. Above zero bullish state = Regime Up. Positive but rolling = Overheating. Bearish mirrors apply on the downside.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def make_line_figure(df: pd.DataFrame, cols: List[str], title: str, zero_line: bool = False) -> go.Figure:
    fig = go.Figure()
    for c in cols:
        if c in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[c], mode="lines", name=c))
    if zero_line:
        fig.add_hline(y=0, line_dash="dot", line_width=1)
    fig.update_layout(
        title=title,
        height=420,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_title="Date",
        template="plotly_dark",
    )
    return fig


def main():
    with st.sidebar:
        st.header("⚙️ Configuration")
        hist_upload = st.file_uploader("Historical ZIP", type=["zip"])
        snap_upload = st.file_uploader("Daily Snapshot CSV", type=["csv"])
        force_rebuild = st.toggle("Force rebuild model", value=False)
        run_backtest = st.button("Run historical backtest")
        reset = st.button("Reset Model")
        st.markdown("---")
        st.subheader("Holistic TSI Settings")
        tsi_long = st.number_input("TSI Long Length", min_value=5, max_value=100, value=25, step=1)
        tsi_short = st.number_input("TSI Short Length", min_value=2, max_value=50, value=13, step=1)
        tsi_signal = st.number_input("TSI Signal Length", min_value=2, max_value=30, value=7, step=1)
        benchmark_symbol = st.selectbox("Backtest Benchmark", ["RSP", "SPY"], index=0)
        bt_mode = st.selectbox("Backtest Mode", ["Long / Cash", "Long / Short"], index=0)
        bt_logic = st.selectbox("Signal Logic", ["Cross vs signal", "Cross zero", "Either"], index=0)
        bt_zero_filter = st.toggle("Require zero filter", value=False)
        bt_min_align = st.slider("Min component alignment %", 0, 100, 50, 5)
        bt_lookback_label = st.selectbox("Backtest Lookback", ["1Y", "2Y", "3Y", "5Y", "10Y", "20Y", "MAX", "Custom"], index=0)
        bt_custom_years = st.number_input("Custom backtest years", min_value=1, max_value=30, value=7, step=1, disabled=(bt_lookback_label != "Custom"))

        if reset:
            for p in [HIST_DAILY_PATH, HIST_WEEKLY_PATH, MODEL_PATH]:
                if p.exists():
                    p.unlink()
            st.success("Model reset")
            st.rerun()

    if bt_lookback_label == "MAX":
        bt_lookback_years = None
    elif bt_lookback_label == "Custom":
        bt_lookback_years = int(bt_custom_years)
    else:
        bt_lookback_years = int(bt_lookback_label.replace("Y", ""))

    model = None
    if hist_upload is not None and (force_rebuild or not MODEL_PATH.exists()):
        with st.spinner("Building historical model..."):
            model = build_model_from_history_bytes(hist_upload.read())
        st.success("Historical model built.")
    if model is None:
        model = load_model()
    if model is None or not HIST_DAILY_PATH.exists():
        st.info("Upload historical ZIP to build the model.")
        return

    daily_feat = pd.read_parquet(HIST_DAILY_PATH)
    snapshot_df = parse_snapshot_csv(snap_upload.read()) if snap_upload is not None else None
    snapshot, prev_snapshot, latest_date = build_snapshot_from_history_and_csv(daily_feat, snapshot_df)

    state_scores = {state: evaluate_state(snapshot, model["states"][state]) for state in ["bounce", "repair", "regime", "fall"]}
    _, _ = score_bands(snapshot, model.get("bands", {}))
    range_df = build_range_map(snapshot, model.get("bands", {}))
    osc_df = compute_repair_oscillator_matrix(snapshot, prev_snapshot)

    canary_hist = pd.DataFrame(model.get("canary_hist", []))
    if not canary_hist.empty:
        canary_hist["date"] = pd.to_datetime(canary_hist["date"])
        row = canary_hist[canary_hist["date"] <= latest_date].tail(1)
        canary_comp = float(row["canary_comp"].iloc[0]) if not row.empty else 0.0
        canary_conf = float(row["canary_conf"].iloc[0]) if not row.empty else 0.0
    else:
        canary_comp, canary_conf = 0.0, 0.0
    canary = {"label": "Risk-On" if canary_comp > 0.05 else "Risk-Off" if canary_comp < -0.05 else "Neutral", "comp": canary_comp, "conf": canary_conf}

    recovery_score = float(np.clip(osc_df["Repair Score"].mean() / 4 * 100, 0, 100)) if not osc_df.empty else np.nan

    cluster_info = model.get("clusters")
    cluster_name, cluster_conf = None, None
    if cluster_info:
        _, cluster_name, cluster_conf = predict_cluster(
            snapshot,
            ClusterArtifacts(
                scaler_mean=cluster_info["mean"],
                scaler_scale=cluster_info["scale"],
                features=cluster_info["features"],
                centroids=cluster_info["centroids"],
                cluster_names=cluster_info["names"],
                silhouette_score=cluster_info.get("silhouette", 0.0),
            ),
        )

    signal = classify_signal(state_scores, canary, recovery_score, cluster_name, osc_df)
    long_setups = nearest_confirmation_from_ranges(snapshot, model.get("bands", {}), bullish=True)
    short_setups = nearest_confirmation_from_ranges(snapshot, model.get("bands", {}), bullish=False)

    holistic_hist, holistic_components = compute_holistic_tsi_history(daily_feat, tsi_long, tsi_short, tsi_signal)
    holistic_row = holistic_hist.loc[holistic_hist.index.max()] if not holistic_hist.empty else pd.Series(dtype=float)
    gauge = holistic_gauge_state(holistic_row)
    holistic_gate = compute_holistic_gate(state_scores, gauge)

    bt_df, bt_stats = (pd.DataFrame(), {})
    if run_backtest and not holistic_hist.empty:
        bt_df, bt_stats = run_holistic_backtest(
            holistic_hist,
            daily_feat,
            benchmark_symbol=benchmark_symbol,
            mode=bt_mode,
            signal_logic=bt_logic,
            require_zero_filter=bt_zero_filter,
            min_align=bt_min_align,
            lookback_years=bt_lookback_years,
        )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_score_card("Bounce", 100 * state_scores["bounce"]["prob"] if pd.notna(state_scores["bounce"]["prob"]) else 0, 100 * state_scores["bounce"]["prob"] if pd.notna(state_scores["bounce"]["prob"]) else np.nan)
    with c2:
        render_score_card("Repair", 100 * state_scores["repair"]["prob"] if pd.notna(state_scores["repair"]["prob"]) else 0, 100 * state_scores["repair"]["prob"] if pd.notna(state_scores["repair"]["prob"]) else np.nan)
    with c3:
        render_score_card("Regime", 100 * state_scores["regime"]["prob"] if pd.notna(state_scores["regime"]["prob"]) else 0, 100 * state_scores["regime"]["prob"] if pd.notna(state_scores["regime"]["prob"]) else np.nan)
    with c4:
        render_score_card("Fall", 100 * state_scores["fall"]["prob"] if pd.notna(state_scores["fall"]["prob"]) else 0, 100 * state_scores["fall"]["prob"] if pd.notna(state_scores["fall"]["prob"]) else np.nan)

    gate_left, gate_right = st.columns([1.15, .85])
    with gate_left:
        render_holistic_gate_card(holistic_gate, state_scores)
    with gate_right:
        render_signal_box(signal["signal"], " | ".join(signal["reasons"][:3]))

    tabs = st.tabs(["Decision Dashboard", "Range Map / State Ladder", "Backtest vs Buy & Hold", "Diagnostics", "Holistic TSI"])
    tab1, tab2, tab3, tab4, tab5 = tabs

    with tab1:
        left, right = st.columns([1.1, 1.2])
        with left:
            render_holistic_gate_card(holistic_gate, state_scores)
            st.markdown("<div class='soft-card'><div class='score-title'>Intraday / Repair Context</div>", unsafe_allow_html=True)
            proxy = proxy_nymo(snapshot, prev_snapshot)
            st.write(f"NYMO Proxy: {fmt_num(proxy['value'])} | Delta: {fmt_num(proxy['delta'])} | State: {proxy['state']}")
            st.write(f"Recovery Score: {fmt_num(recovery_score,1)} | Canary: {canary['label']} | Cluster: {cluster_name or 'n/a'}")
            st.markdown("</div>", unsafe_allow_html=True)
            render_holistic_gauge(gauge, holistic_row)
        with right:
            st.markdown("<div class='soft-card'><div class='score-title'>Hold Trade Setup Parameters</div>", unsafe_allow_html=True)
            render_setup_lines(long_setups, "Go LONG if these start to trigger:")
            render_setup_lines(short_setups, "Go SHORT if these start to trigger:")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='soft-card'><div class='score-title'>Repair Oscillator Matrix</div>", unsafe_allow_html=True)
        st.dataframe(osc_df, width='stretch', hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("<div class='soft-card'><div class='score-title'>Range Map / State Ladder</div>", unsafe_allow_html=True)
        if not range_df.empty:
            summary = range_df["State Ladder"].value_counts().rename_axis("State").reset_index(name="Count")
            st.dataframe(summary, width='stretch', hide_index=True)
            st.dataframe(range_df, width='stretch', hide_index=True)
        else:
            st.write("No range map available.")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        st.markdown("<div class='soft-card'><div class='score-title'>Backtest vs Buy & Hold</div>", unsafe_allow_html=True)
        st.markdown(
            f"<span class='pill pill-blue'>TSI params: {tsi_long},{tsi_short},{tsi_signal}</span>"
            f"<span class='pill pill-green'>Benchmark: {benchmark_symbol}</span>"
            f"<span class='pill pill-yellow'>Mode: {bt_mode}</span>"
            f"<span class='pill pill-orange'>Lookback: {bt_lookback_label if bt_lookback_label != 'Custom' else str(bt_custom_years)+'Y'}</span>",
            unsafe_allow_html=True,
        )
        if not run_backtest:
            st.info("Click 'Run historical backtest' in the sidebar to generate the equity curve and lead/lag study.")
        elif bt_df.empty:
            st.warning("Backtest could not run. Check the uploaded history for the selected benchmark and required series.")
        else:
            a, b, c, d = st.columns(4)
            with a:
                st.metric("Strategy Return", f"{bt_stats['strategy_return']*100:.1f}%")
            with b:
                st.metric("Buy & Hold Return", f"{bt_stats['buy_hold_return']*100:.1f}%")
            with c:
                st.metric("Strategy Max DD", f"{bt_stats['strategy_max_dd']*100:.1f}%")
            with d:
                st.metric("Trades", bt_stats["trades"])

            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(x=bt_df.index, y=bt_df["strategy_equity"], mode="lines", name="Holistic TSI Strategy"))
            fig_eq.add_trace(go.Scatter(x=bt_df.index, y=bt_df["buy_hold_equity"], mode="lines", name=f"{benchmark_symbol} Buy & Hold"))
            fig_eq.update_layout(height=430, title="Equity Curve", template="plotly_dark", margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_eq, use_container_width=True)

            overlay = pd.DataFrame(index=bt_df.index)
            overlay[f"{benchmark_symbol}_price"] = bt_df["price"]
            overlay["holistic_tsi"] = bt_df["holistic_tsi"]
            overlay["holistic_signal"] = bt_df["holistic_signal"]
            overlay["price_tsi_25_13_7"] = bt_df["price_tsi"]
            overlay["price_tsi_signal_25_13_7"] = bt_df["price_tsi_signal"]
            fig_overlay = go.Figure()
            fig_overlay.add_trace(go.Scatter(x=overlay.index, y=overlay[f"{benchmark_symbol}_price"], mode="lines", name=f"{benchmark_symbol} Price", yaxis="y1"))
            fig_overlay.add_trace(go.Scatter(x=overlay.index, y=overlay["holistic_tsi"], mode="lines", name="Holistic TSI", yaxis="y2"))
            fig_overlay.add_trace(go.Scatter(x=overlay.index, y=overlay["holistic_signal"], mode="lines", name="Holistic Signal", yaxis="y2"))
            fig_overlay.add_trace(go.Scatter(x=overlay.index, y=overlay["price_tsi_25_13_7"], mode="lines", name=f"{benchmark_symbol} Price TSI", yaxis="y2"))
            fig_overlay.add_trace(go.Scatter(x=overlay.index, y=overlay["price_tsi_signal_25_13_7"], mode="lines", name=f"{benchmark_symbol} Price TSI Signal", yaxis="y2"))
            fig_overlay.update_layout(
                title=f"Holistic TSI vs {benchmark_symbol} Price / Price TSI",
                template="plotly_dark",
                height=480,
                margin=dict(l=20, r=20, t=50, b=20),
                yaxis=dict(title="Price"),
                yaxis2=dict(title="TSI", overlaying="y", side="right"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            fig_overlay.add_hline(y=0, line_dash="dot", line_width=1, yref="y2")
            st.plotly_chart(fig_overlay, use_container_width=True)

            lead_summary = bt_stats.get("lead_summary", {})
            if lead_summary:
                rows = []
                for k, v in lead_summary.items():
                    rows.append(
                        {
                            "Event": k,
                            "Holistic Cross Lead (days)": v.get("holistic_cross_lead_days"),
                            "Holistic Zero Lead (days)": v.get("holistic_zero_lead_days"),
                            f"{benchmark_symbol} Price TSI Turn Lead (days)": v.get("benchmark_tsi_turn_lead_days"),
                            "Samples": v.get("count"),
                        }
                    )
                st.markdown("**Lead / Lag Event Study**")
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab4:
        st.markdown("<div class='soft-card'><div class='score-title'>Diagnostics</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Cluster Diagnostics**")
            st.write(f"Cluster: {cluster_name or 'n/a'}")
            st.write(f"Cluster Confidence: {fmt_num(cluster_conf,2)}")
            st.write(f"Canary Composite: {fmt_num(canary['comp'],2)}")
            st.write(f"Canary Confidence: {fmt_num(canary['conf'],1)}")
        with col2:
            st.markdown("**Model Stats**")
            st.write(f"Historical rows: {model.get('meta', {}).get('rows', 'n/a')}")
            st.write(f"Latest date: {latest_date.date()}")
            st.write(f"Repair breadth: {int((osc_df['Repair Score'] >= 2).sum())}/{len(osc_df) if not osc_df.empty else 0}")

        if model.get("clusters", {}).get("stats"):
            st.markdown("**Historical Cluster Summary**")
            st.dataframe(pd.DataFrame(model["clusters"]["stats"]), width="stretch", hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab5:
        st.markdown("<div class='soft-card'><div class='score-title'>Holistic TSI</div>", unsafe_allow_html=True)
        if holistic_hist.empty:
            st.warning("Holistic TSI could not be built from the uploaded history.")
        else:
            top_left, top_mid, top_right = st.columns([1.1, 1, 1])
            with top_left:
                render_holistic_gauge(gauge, holistic_row)
            with top_mid:
                st.markdown("<div class='kpi-box'>", unsafe_allow_html=True)
                st.markdown("<div class='score-title'>TSI State Context</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='score-value-sm'>TSI {fmt_num(holistic_row.get('holistic_tsi', np.nan),2)}</div>", unsafe_allow_html=True)
                st.markdown(f"<span class='pill pill-blue'>Signal {fmt_num(holistic_row.get('holistic_signal', np.nan),2)}</span>", unsafe_allow_html=True)
                st.markdown(f"<span class='pill pill-green'>Above Signal {fmt_num(holistic_row.get('pct_above_signal', np.nan),1)}%</span>", unsafe_allow_html=True)
                st.markdown(f"<span class='pill pill-yellow'>Above Zero {fmt_num(holistic_row.get('pct_above_zero', np.nan),1)}%</span>", unsafe_allow_html=True)
                st.markdown(f"<span class='pill pill-blue'>Slope(3) {fmt_num(holistic_row.get('slope3', np.nan),2)}</span>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with top_right:
                st.markdown("<div class='kpi-box'>", unsafe_allow_html=True)
                st.markdown("<div class='score-title'>Bucket Readout</div>", unsafe_allow_html=True)
                for bucket in ["breadth", "leadership", "risk"]:
                    bt_tsi = holistic_row.get(f"{bucket}_tsi", np.nan)
                    bt_sig = holistic_row.get(f"{bucket}_signal", np.nan)
                    pct = holistic_row.get(f"{bucket}_pct_above_signal", np.nan)
                    st.markdown(f"<div class='tiny-muted'>{bucket.title()}</div>", unsafe_allow_html=True)
                    st.markdown(f"<span class='pill pill-blue'>TSI {fmt_num(bt_tsi,2)}</span><span class='pill pill-yellow'>Signal {fmt_num(bt_sig,2)}</span><span class='pill pill-green'>Aligned {fmt_num(pct,1)}%</span>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            fig_h = make_line_figure(
                holistic_hist[["holistic_tsi", "holistic_signal", "breadth_tsi", "leadership_tsi", "risk_tsi"]].dropna(how="all"),
                ["holistic_tsi", "holistic_signal", "breadth_tsi", "leadership_tsi", "risk_tsi"],
                "Holistic TSI and Bucket TSIs",
                zero_line=True,
            )
            st.plotly_chart(fig_h, use_container_width=True)

            piv = daily_feat.pivot(index="date", columns="symbol", values="close").sort_index()
            bench_for_overlay = benchmark_symbol if benchmark_symbol in piv.columns else "RSP"
            overlay = holistic_hist.join(piv[[bench_for_overlay]].rename(columns={bench_for_overlay: "price"}), how="left").dropna(subset=["price"])
            price_tsi, price_sig = true_strength_index(overlay["price"], tsi_long, tsi_short, tsi_signal)
            fig_overlay = go.Figure()
            fig_overlay.add_trace(go.Scatter(x=overlay.index, y=overlay["price"], mode="lines", name=f"{bench_for_overlay} Price", yaxis="y1"))
            fig_overlay.add_trace(go.Scatter(x=overlay.index, y=overlay["holistic_tsi"], mode="lines", name="Holistic TSI", yaxis="y2"))
            fig_overlay.add_trace(go.Scatter(x=overlay.index, y=overlay["holistic_signal"], mode="lines", name="Holistic Signal", yaxis="y2"))
            fig_overlay.add_trace(go.Scatter(x=overlay.index, y=price_tsi, mode="lines", name=f"{bench_for_overlay} Price TSI", yaxis="y2"))
            fig_overlay.add_trace(go.Scatter(x=overlay.index, y=price_sig, mode="lines", name=f"{bench_for_overlay} Price TSI Signal", yaxis="y2"))
            fig_overlay.update_layout(
                title=f"Overlay: Holistic TSI vs {bench_for_overlay} Price and Price TSI",
                template="plotly_dark",
                height=500,
                margin=dict(l=20, r=20, t=50, b=20),
                yaxis=dict(title="Price"),
                yaxis2=dict(title="TSI", overlaying="y", side="right"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            fig_overlay.add_hline(y=0, line_dash="dot", line_width=1, yref="y2")
            st.plotly_chart(fig_overlay, use_container_width=True)

            if not holistic_components.empty:
                latest_comp = holistic_components[holistic_components["date"] == holistic_components["date"].max()].copy()
                latest_comp["state"] = np.select(
                    [
                        (latest_comp["tsi"] > latest_comp["signal"]) & (latest_comp["tsi"] > 0),
                        (latest_comp["tsi"] > latest_comp["signal"]) & (latest_comp["tsi"] <= 0),
                        (latest_comp["tsi"] <= latest_comp["signal"]) & (latest_comp["tsi"] > 0),
                        (latest_comp["tsi"] <= latest_comp["signal"]) & (latest_comp["tsi"] <= 0),
                    ],
                    [
                        "Bullish regime",
                        "Repair / bounce",
                        "Weakening",
                        "Bearish regime",
                    ],
                    default="Mixed",
                )
                latest_comp = latest_comp.sort_values(["bucket", "gap"], ascending=[True, False])
                st.markdown("**Latest Component Readout**")
                st.dataframe(
                    latest_comp[["bucket", "symbol", "weight", "tsi", "signal", "gap", "slope3", "state"]].rename(
                        columns={"bucket": "Bucket", "symbol": "Symbol", "weight": "Weight", "tsi": "TSI", "signal": "Signal", "gap": "Gap", "slope3": "Slope(3)", "state": "State"}
                    ),
                    width="stretch",
                    hide_index=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
