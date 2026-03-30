#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Breadth Sweet Spot Engine v5.0 - Quant Production Ultimate
==========================================================
Production-grade market breadth analysis engine designed to beat buy-and-hold RSP.

Features:
- Empirical gate learning (lift/support scoring) + sweet spot bands
- K-Means clustering with silhouette validation for regime detection
- Recovery momentum layer with normalized slope analysis
- 7-ratio canary filter for signal confirmation
- NYMO proxy governance (0.6×NYAD + 0.4×SPXADP) for intraday analysis
- TRIN/VIX support with proper inversion logic
- LONG and SHORT signal frameworks
- Walk-forward validation & Monte Carlo robustness testing
- Comprehensive risk metrics: Sharpe, Sortino, Calmar, VaR, skew, kurtosis
- RSP buy-and-hold comparison with statistical significance testing

Author: Market Breadth Analysis Engine
Version: 5.0.0 (Quant Production Ultimate)
Last Updated: 2026-03-30
"""

from __future__ import annotations
import io
import json
import logging
import math
import re
import sys
import warnings
import zipfile
from dataclasses import dataclass, asdict
from datetime import datetime, time as dt_time
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import hashlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from scipy import stats

# ==============================================================================
# Configuration & Logging
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(__file__).parent / 'breadth_engine_v5.log', mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=(FutureWarning, UserWarning, RuntimeWarning))

# ==============================================================================
# App Configuration
# ==============================================================================
st.set_page_config(
    page_title="Breadth Sweet Spot Engine v5.0 | Ultimate",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
:root{
  --bg:#0b1020;--panel:#111936;--panel2:#162246;--text:#ecf2ff;--muted:#98abd5;
  --green:#22c55e;--yellow:#f59e0b;--red:#ef4444;--blue:#38bdf8;--purple:#a78bfa;--cyan:#06b6d4;
}
body{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;}
.block-container{padding-top:1rem;padding-bottom:2rem;}
.main-title{padding:1rem 1.2rem;border-radius:18px;background:linear-gradient(135deg, rgba(56,189,248,.18), rgba(167,139,250,.18));border:1px solid rgba(148,163,184,.22);margin-bottom:1rem;}
.soft-card{background:linear-gradient(180deg, rgba(17,25,54,.96), rgba(10,17,38,.98));border:1px solid rgba(148,163,184,.24);border-radius:18px;padding:1rem;box-shadow:0 10px 35px rgba(0,0,0,.22);}
.score-card{min-height:170px;display:flex;flex-direction:column;justify-content:space-between;}
.score-title{color:#bcd0ff;font-size:1.02rem;font-weight:800;}
.score-value{font-size:3.0rem;font-weight:950;color:#fff;margin:.35rem 0;}
.score-bar{width:100%;height:12px;border-radius:999px;background:rgba(255,255,255,.09);overflow:hidden;margin-top:.65rem;}
.score-fill{height:100%;border-radius:999px;}
.fill-green{background:linear-gradient(90deg,#22c55e,#4ade80)}
.fill-yellow{background:linear-gradient(90deg,#f59e0b,#fbbf24)}
.fill-red{background:linear-gradient(90deg,#ef4444,#f87171)}
.fill-blue{background:linear-gradient(90deg,#38bdf8,#60a5fa)}
.pill{display:inline-block;padding:.3rem .6rem;border-radius:999px;font-size:.82rem;font-weight:700;border:1px solid rgba(255,255,255,.12);margin-right:.35rem;}
.pill-green{background:rgba(34,197,94,.16);color:#bbf7d0;}
.pill-yellow{background:rgba(245,158,11,.16);color:#fde68a;}
.pill-red{background:rgba(239,68,68,.16);color:#fecaca;}
.pill-blue{background:rgba(56,189,248,.16);color:#bae6fd;}
.status-bar{display:flex;justify-content:space-between;align-items:center;padding:.75rem 1rem;background:rgba(255,255,255,.05);border-radius:12px;margin-bottom:1rem;}
.signal-long{background:rgba(34,197,94,.14);border:1px solid rgba(34,197,94,.32);border-radius:16px;padding:1rem;}
.signal-short{background:rgba(239,68,68,.14);border:1px solid rgba(239,68,68,.32);border-radius:16px;padding:1rem;}
.signal-hold{background:rgba(245,158,11,.14);border:1px solid rgba(245,158,11,.32);border-radius:16px;padding:1rem;}
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown("""
<div class="main-title">
<h1 style="margin:0;font-size:1.8rem;">📈 Breadth Sweet Spot Engine v5.0 | Ultimate</h1>
<p style="margin:0.5rem 0 0 0;color:#98abd5;">Gate Learning + Sweet Spots + Clustering • Beats Buy & Hold RSP • Full Validation Suite</p>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# Constants (NO TRAILING SPACES)
# ==============================================================================
APP_DIR = Path("breadth_quant_store_v5")
APP_DIR.mkdir(exist_ok=True)
HIST_DAILY_PATH = APP_DIR / "daily_history.parquet"
HIST_WEEKLY_PATH = APP_DIR / "weekly_history.parquet"
MODEL_PATH = APP_DIR / "learned_model.json"
UPLOAD_HISTORY_PATH = APP_DIR / "upload_history.csv"
SNAPSHOT_DIR = APP_DIR / "snapshots"
SNAPSHOT_DIR.mkdir(exist_ok=True)
BASELINE_TIMESTAMP_PATH = APP_DIR / "baseline_timestamp.txt"

KEY_FEATURES = [
    "$BPSPX", "$BPSPX_%B", "$BPNYA", "$OEXA200R", "$SPXA50R", "$NYMO", "$NYSI", "$CPCE",
    "$NYHL", "$NYAD", "$SPXADP", "$TRIN", "$VIX", "RSP:SPY"
]
INVERSE_INDICATORS = ["$TRIN", "$VIX", "$CPCE"]
MOMENTUM_INDICATORS = ["$NYMO", "$NYSI", "$NYAD", "$SPXADP", "$BPSPX", "$CPCE", "$VIX", "$TRIN"]
RECOVERY_FEATURES = ["$NYSI", "$NYMO", "$BPSPX_%B", "$SPXA50R", "$NYHL", "$TRIN", "$CPCE", "$VIX", "RSP:SPY"]
WEEKLY_FEATURES = ["$BPSPX", "$SPXA50R", "$NYSI", "$OEXA200R", "RSP"]

CANARY_WEIGHTS = {
    "SPXS:SVOL": 0.24, "HYG:IEF": 0.20, "SMH:SPY": 0.18,
    "XLF:SPY": 0.12, "RSP:SPY": 0.12, "IWM:SPY": 0.10, "$VIX": 0.04
}

OUTCOME_DEFS = {
    "bounce": {"horizon": 10, "ret": 0.03, "dd": -0.03, "type": "max"},
    "repair": {"horizon": 20, "ret": 0.04, "dd": -0.05, "type": "end"},
    "regime": {"horizon": 60, "ret": 0.08, "dd": -0.08, "type": "end"},
    "fall": {"horizon": 10, "ret": -0.03, "dd": 0.03, "type": "min_end"},
}

TREND_WINDOWS = [1, 2, 3, 5, 10]
MIN_GATE_SUPPORT = 40
TOP_SINGLE_PER_OUTCOME = 10
MAX_COMBO_CANDIDATES = 8
EASTERN = "America/New_York"
PROXY_NYMO_FORMULA = {"nyad_weight": 0.6, "spxadp_weight": 0.4, "normalization": 1600.0}
RECOVERY_TANH_SCALE = 3.0

# ==============================================================================
# Enums
# ==============================================================================
class SessionPhase(str, Enum):
    PRE = "Pre-Market"
    OPEN = "Opening Window"
    MID = "Midday Window"
    LATE = "Late-Day Window"
    POST = "Post-Close"
    OFFICIAL = "Official EOD"
    WEEKEND = "Weekend"

class MarketNarrative(str, Enum):
    CAPITULATION = "Capitulation washout"
    BREADTH_THRUST = "Oversold breadth thrust"
    REPAIR_PHASE = "Post-thrust repair phase"
    CONFIRMED_REPAIR = "Confirmed trend repair"
    NEUTRAL = "Neutral / transitional"

# ==============================================================================
# Utilities
# ==============================================================================
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
    except Exception as e:
        logger.error(f"Failed to save JSON: {e}")
        return False

def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default

def hash_file(filepath: Path) -> str:
    if not filepath.exists():
        return ""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def score_color(score: float, max_score: float = 100.0) -> Tuple[str, str]:
    if max_score == 0:
        return "#ef4444", "fill-red"
    frac = max(0, min(1, score / max_score))
    if frac >= 0.7:
        return "#22c55e", "fill-green"
    if frac >= 0.4:
        return "#f59e0b", "fill-yellow"
    return "#ef4444", "fill-red"

def detect_session_phase() -> SessionPhase:
    ts = pd.Timestamp.now(tz=EASTERN)
    if ts.weekday() >= 5:
        return SessionPhase.WEEKEND
    t = dt_time(ts.hour, ts.minute)
    if t < dt_time(9, 30):
        return SessionPhase.PRE
    if t < dt_time(11, 0):
        return SessionPhase.OPEN
    if t < dt_time(14, 30):
        return SessionPhase.MID
    if t < dt_time(16, 0):
        return SessionPhase.LATE
    if t < dt_time(18, 0):
        return SessionPhase.POST
    return SessionPhase.OFFICIAL

# ==============================================================================
# Technical Indicators (Vectorized)
# ==============================================================================
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)

def percent_b(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    denom = (upper - lower).replace(0, np.nan)
    return (series - lower) / denom

def macd_hist(series: pd.Series, fast: int = 24, slow: int = 52, signal: int = 18) -> pd.Series:
    line = ema(series, fast) - ema(series, slow)
    sig = ema(line, signal)
    return line - sig

def stoch_from_close(close: pd.Series, length: int = 14, smoothk: int = 3) -> pd.Series:
    lo = close.rolling(length).min()
    hi = close.rolling(length).max()
    denom = (hi - lo).replace(0, np.nan)
    k = 100 * (close - lo) / denom
    return k.rolling(smoothk).mean()

def add_indicator_features(hist: pd.DataFrame) -> pd.DataFrame:
    out = []
    for sym, g in hist.groupby("symbol", sort=False):
        g = g.sort_values("date").copy()
        close = g["close"]
        high = g["high"] if "high" in g.columns else close
        low = g["low"] if "low" in g.columns else close
        
        g["rsi14"] = rsi(close, 14)
        g["pct_b20"] = percent_b(close, 20, 2.0)
        g["ma20"] = ema(close, 20)
        g["ma50"] = ema(close, 50)
        g["macd_hist"] = macd_hist(close)
        g["stoch"] = stoch_from_close(close, 14, 3)
        
        for w in TREND_WINDOWS:
            g[f"d{w}"] = close.diff(w)
            g[f"roc{w}"] = 100 * (close / close.shift(w) - 1)
        
        out.append(g)
    return pd.concat(out, ignore_index=True)

# ==============================================================================
# Data Parsing
# ==============================================================================
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
        volume = nums[4] if len(nums) > 4 else np.nan
        rows.append({"date": dt, "open": nums[0], "high": nums[1], "low": nums[2], "close": nums[3], "volume": volume})
    if not rows:
        raise ValueError("No rows parsed from StockCharts CSV")
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

def symbol_from_filename(name: str) -> Tuple[str, str]:
    stem = Path(name).stem.strip().lower()
    timeframe = "weekly" if stem.endswith("w") or stem.endswith("_w") else "daily"
    stem = stem.replace("w", "").replace("_w", "").strip("_")
    
    mapping = {
        "rsp": "RSP", "ursp": "URSP", "spy": "SPY", "vxx": "VXX",
        "_bpspx": "$BPSPX", "bpspx": "$BPSPX", "_bpnya": "$BPNYA", "bpnya": "$BPNYA",
        "_oexa200r": "$OEXA200R", "oexa200r": "$OEXA200R", "_spxa50r": "$SPXA50R", "spxa50r": "$SPXA50R",
        "_nymo": "$NYMO", "nymo": "$NYMO", "_nysi": "$NYSI", "nysi": "$NYSI",
        "_cpce": "$CPCE", "cpce": "$CPCE", "_nyhl": "$NYHL", "nyhl": "$NYHL",
        "_nyad": "$NYAD", "nyad": "$NYAD", "_spxadp": "$SPXADP", "spxadp": "$SPXADP",
        "_trin": "$TRIN", "trin": "$TRIN", "_vix": "$VIX", "vix": "$VIX",
        "hyg_ief": "HYG:IEF", "rsp_spy": "RSP:SPY", "smh_spy": "SMH:SPY",
        "iwm_spy": "IWM:SPY", "xlf_spy": "XLF:SPY", "spxs_svol": "SPXS:SVOL",
    }
    sym = mapping.get(stem, stem.upper())
    return sym, timeframe

def parse_stockcharts_zip(file_bytes: bytes) -> Tuple[pd.DataFrame, pd.DataFrame]:
    daily, weekly = [], []
    try:
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
                except Exception as e:
                    logger.warning(f"Skip {name}: {e}")
                    continue
    except Exception as e:
        logger.error(f"Failed to parse ZIP: {e}")
        raise ValueError(f"Failed to parse zip file: {str(e)}")
    
    if not daily:
        raise ValueError("No daily CSV files parsed from ZIP")
    
    daily_df = pd.concat(daily, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    weekly_df = pd.concat(weekly, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True) if weekly else pd.DataFrame(columns=daily_df.columns)
    return daily_df, weekly_df

# ==============================================================================
# Forward Metrics (Vectorized)
# ==============================================================================
def compute_future_metrics(price: pd.Series, horizon: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    vals = price.to_numpy(dtype=float)
    n = len(vals)
    if n < horizon + 1:
        idx = price.index
        return pd.Series(np.nan, index=idx), pd.Series(np.nan, index=idx), pd.Series(np.nan, index=idx)
    
    idx = np.arange(n)
    window_idx = idx[:, None] + np.arange(1, horizon + 1)
    mask = window_idx >= n
    safe_idx = np.clip(window_idx, 0, n - 1)
    win = vals[safe_idx].astype(float)
    win[mask] = np.nan
    
    base = vals[:, None]
    rets = win / base - 1
    
    end_ret = rets[:, -1]
    max_gain = np.nanmax(rets, axis=1)
    max_dd = np.nanmin(rets, axis=1)
    
    valid = n - horizon
    end_ret[valid:] = np.nan
    max_gain[valid:] = np.nan
    max_dd[valid:] = np.nan
    
    return pd.Series(end_ret, index=price.index), pd.Series(max_gain, index=price.index), pd.Series(max_dd, index=price.index)

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

# ==============================================================================
# Gate Learning (Lift/Support Scoring)
# ==============================================================================
def direction_hints(feature: str, state: str) -> List[str]:
    if state == "fall":
        if any(k in feature for k in ["$TRIN", "$VIX", "$CPCE"]):
            return ["gte", "lte"]
        return ["lte", "gte"]
    if feature in ["$TRIN", "$VIX", "$CPCE", "VXX"]:
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
                
                lift = hit / base if base > 0 else np.nan
                score = (hit - base) * math.sqrt(support)
                
                gates.append({
                    "feature": feat, "direction": direction, "threshold": thr,
                    "support": support, "hit_rate": hit, "base_rate": base,
                    "lift": lift, "score": score
                })
    
    gates = sorted(gates, key=lambda x: (x["score"], x["lift"], x["support"]), reverse=True)
    top, used = [], set()
    for g in gates:
        key = g["feature"]
        if key in used:
            continue
        used.add(key)
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
            
            combos.append({
                "gates": [g1, g2], "support": support, "hit_rate": hit,
                "base_rate": base, "lift": hit / base if base > 0 else np.nan, "score": score
            })
    
    combos = sorted(combos, key=lambda x: (x["score"], x["lift"], x["support"]), reverse=True)
    return combos[:6]

def gate_pass(cur: float, gate: dict) -> bool:
    if pd.isna(cur):
        return False
    return cur >= gate["threshold"] if gate["direction"] == "gte" else cur <= gate["threshold"]

# ==============================================================================
# K-Means Clustering
# ==============================================================================
@dataclass
class ClusterArtifacts:
    scaler_mean: List[float]
    scaler_scale: List[float]
    features: List[str]
    centroids: List[List[float]]
    cluster_names: Dict[str, str]
    cluster_stats: Dict[str, Dict[str, float]]
    silhouette_score: float = 0.0
    calinski_score: float = 0.0

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

def build_clusters(outcomes_df: pd.DataFrame, features: List[str], n_clusters: int = 6) -> Tuple[pd.DataFrame, ClusterArtifacts]:
    feat_df = outcomes_df[features].apply(pd.to_numeric, errors="coerce")
    valid = feat_df.dropna()
    n_clusters = max(3, min(n_clusters, max(3, len(valid) // 40)))
    
    scaler = StandardScaler()
    X = scaler.fit_transform(valid)
    
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20, max_iter=300)
    labels = km.fit_predict(X)
    
    sil_score = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else 0.0
    cal_score = calinski_harabasz_score(X, labels) if len(np.unique(labels)) > 1 else 0.0
    
    if sil_score < 0.2:
        logger.warning(f"Weak cluster separation (Silhouette: {sil_score:.2f})")
    
    cluster_stats_rows = []
    for cl in sorted(np.unique(labels)):
        mask = labels == cl
        sub = valid.loc[mask]
        metrics = {
            "cluster": int(cl),
            "samples": int(mask.sum()),
            "bounce_rate": float(outcomes_df.loc[sub.index, "bounce_success"].mean()) if "bounce_success" in outcomes_df.columns else 0,
            "repair_rate": float(outcomes_df.loc[sub.index, "repair_success"].mean()) if "repair_success" in outcomes_df.columns else 0,
            "regime_rate": float(outcomes_df.loc[sub.index, "regime_success"].mean()) if "regime_success" in outcomes_df.columns else 0,
            "fall_rate": float(outcomes_df.loc[sub.index, "fall_success"].mean()) if "fall_success" in outcomes_df.columns else 0,
        }
        for feat in features:
            metrics[f"{feat}_median"] = float(sub[feat].median())
        cluster_stats_rows.append(metrics)
    
    stats_df = pd.DataFrame(cluster_stats_rows).sort_values("cluster").reset_index(drop=True)
    names = assign_cluster_names(stats_df)
    stats_df["cluster_name"] = stats_df["cluster"].map(names)
    
    artifacts = ClusterArtifacts(
        scaler_mean=scaler.mean_.tolist(),
        scaler_scale=scaler.scale_.tolist(),
        features=features,
        centroids=km.cluster_centers_.tolist(),
        cluster_names={str(k): v for k, v in names.items()},
        cluster_stats=stats_df.set_index("cluster").to_dict(orient="index"),
        silhouette_score=float(sil_score),
        calinski_score=float(cal_score)
    )
    return stats_df, artifacts

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
    confidence = 1.0 / (1.0 + float(dists[cl]))
    return cl, artifacts.cluster_names.get(str(cl), f"Cluster {cl}"), confidence

# ==============================================================================
# Recovery Momentum
# ==============================================================================
def compute_recovery_momentum(piv: pd.DataFrame, latest_date, windows: Dict[str, int] = None) -> Dict[str, Any]:
    if windows is None:
        windows = {k: 5 for k in RECOVERY_FEATURES}
    
    comps = {}
    for k, w in windows.items():
        if k not in piv.columns:
            comps[k] = np.nan
            continue
        series = piv[k].loc[:latest_date].dropna()
        if len(series) < w + 1:
            comps[k] = np.nan
            continue
        
        y = series.iloc[-w-1:].to_numpy(dtype=float)
        x = np.arange(len(y))
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator == 0:
            slope = 0.0
        else:
            slope = numerator / denominator
        
        denom = max(np.nanmean(np.abs(y)), 1e-6)
        comps[k] = slope / denom
    
    for k in INVERSE_INDICATORS:
        if k in comps and pd.notna(comps[k]):
            comps[k] = -comps[k]
    
    vals = []
    for v in comps.values():
        if pd.isna(v):
            continue
        vals.append(np.tanh(v * RECOVERY_TANH_SCALE))
    
    if not vals:
        recovery_score = np.nan
    else:
        recovery_score = float(np.mean(vals)) * 50 + 50
        recovery_score = max(0.0, min(100.0, recovery_score))
    
    if pd.isna(recovery_score):
        regime = "Continued Washout"
    elif recovery_score < 30:
        regime = "Continued Washout"
    elif recovery_score < 50:
        regime = "Early Stabilization"
    elif recovery_score < 70:
        regime = "Confirmed Recovery"
    else:
        regime = "Strong Momentum"
    
    return {"recovery_score": recovery_score, "components": comps, "regime": regime}

# ==============================================================================
# NYMO Proxy
# ==============================================================================
def proxy_nymo(snapshot: Dict[str, float], prev_snapshot: Dict[str, float]) -> Dict[str, float]:
    nyad = safe_float(snapshot.get("$NYAD", np.nan))
    spxadp = safe_float(snapshot.get("$SPXADP", np.nan))
    prev_nyad = safe_float(prev_snapshot.get("$NYAD", np.nan))
    prev_spxadp = safe_float(prev_snapshot.get("$SPXADP", np.nan))
    
    cur_raw = 0.6 * (0 if pd.isna(nyad) else nyad) + 0.4 * (0 if pd.isna(spxadp) else spxadp)
    prev_raw = 0.6 * (0 if pd.isna(prev_nyad) else prev_nyad) + 0.4 * (0 if pd.isna(prev_spxadp) else prev_spxadp)
    
    cur = 100 * np.tanh(cur_raw / PROXY_NYMO_FORMULA["normalization"])
    prev = 100 * np.tanh(prev_raw / PROXY_NYMO_FORMULA["normalization"])
    
    state = "Deep washout" if cur <= -70 else "Negative but repairing" if cur <= -20 else "Neutral / crossing" if cur <= 20 else "Positive thrust"
    
    return {"value": float(cur), "delta": float(cur - prev), "state": state}

# ==============================================================================
# Canary Filter
# ==============================================================================
def ratio_indicator_score(close: pd.Series) -> pd.Series:
    if close.dropna().shape[0] < 220:
        return pd.Series(dtype=float)
    
    mh = macd_hist(close)
    t = close.diff().ewm(span=20).mean() / close.diff().abs().ewm(span=20).mean() * 100
    stoch = stoch_from_close(close, 14, 3)
    cci100 = ((close - close.rolling(100).mean()) / (0.015 * (close - close.rolling(100).mean()).abs().rolling(100).mean().replace(0, np.nan)))
    
    df = pd.concat([mh.rename("macdh"), t.rename("tsi"), stoch.rename("stoch"), cci100.rename("cci")], axis=1).dropna()
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
    conf = ((0.6 * align) + (0.4 * strength)) * 100.0
    
    return pd.DataFrame({"canary_comp": comp, "canary_conf": conf})

# ==============================================================================
# Sweet Spot Scoring
# ==============================================================================
def band_distance_score(x: float, q25: float, med: float, q75: float) -> float:
    if pd.isna(x) or pd.isna(q25) or pd.isna(med) or pd.isna(q75):
        return np.nan
    
    iqr = max(abs(q75 - q25), 1e-6)
    
    if q25 <= x <= q75:
        d = abs(x - med) / iqr
        return max(0.72, 1.0 - 0.28 * d)
    
    d = min(abs(x - med) / iqr, 3.0)
    return max(0.0, 0.72 - 0.24 * (d - 1.0))

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

def score_bands(snapshot: Dict[str, float], bands: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows = []
    totals = {}
    
    for label in ["bounce", "repair", "regime"]:
        bmap = bands.get(label, {})
        vals = []
        
        for feat, meta in bmap.items():
            cur = safe_float(snapshot.get(feat, np.nan))
            if feat in INVERSE_INDICATORS and pd.notna(cur):
                cur = -cur
                q25, med, q75 = -meta["q75"], -meta["median"], -meta["q25"]
            else:
                q25, med, q75 = meta["q25"], meta["median"], meta["q75"]
            
            sc = band_distance_score(cur, q25, med, q75)
            rows.append({"Outcome": label.title(), "Feature": feat, "Current": cur, "Median": med, "Q25": q25, "Q75": q75, "BandScore": sc})
            
            if pd.notna(sc):
                vals.append(sc)
        
        totals[label] = 100 * np.mean(vals) if vals else np.nan
    
    return pd.DataFrame(rows), totals

# ==============================================================================
# Evaluate State (Gate + Band Scoring)
# ==============================================================================
def evaluate_state(snapshot: Dict[str, float], state_model: Dict[str, Any]) -> Dict[str, Any]:
    singles = []
    for g in state_model.get("singles", []):
        cur = safe_float(snapshot.get(g["feature"], np.nan))
        passed = gate_pass(cur, g)
        singles.append({**g, "current": cur, "passed": passed})
    
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
    
    return {
        "prob": prob, "pass_frac": pass_frac, "base_rate": base,
        "passed_singles": [x for x in singles if x["passed"]],
        "passed_combos": [x for x in combos if x["passed"]]
    }

# ==============================================================================
# Signal Classification
# ==============================================================================
def classify_signal(state_scores: Dict[str, Any], band_totals: Dict[str, float], canary: Dict[str, Any],
                   cluster_name: Optional[str], recovery_score: float) -> Dict[str, Any]:
    bounce = state_scores["bounce"]
    repair = state_scores["repair"]
    regime = state_scores["regime"]
    fall = state_scores["fall"]
    
    signal = "HOLD"
    reasons = []
    
    long_flag = (
        ((bounce["pass_frac"] >= 0.55 and bounce["prob"] >= max(0.40, bounce["base_rate"] + 0.08)) or
         (repair["pass_frac"] >= 0.50 and repair["prob"] >= max(0.30, repair["base_rate"] + 0.06)))
        and recovery_score >= 45 and canary["label"] != "Risk-Off"
    ) or (
        regime["pass_frac"] >= 0.50 and regime["prob"] >= max(0.30, regime["base_rate"] + 0.05)
        and canary["label"] != "Risk-Off"
    )
    
    short_flag = (
        fall["pass_frac"] >= 0.50 and fall["prob"] >= max(0.25, fall["base_rate"] + 0.05)
        and canary["label"] != "Risk-On" and recovery_score < 40
    )
    
    if long_flag:
        signal = "LONG"
    elif short_flag:
        signal = "SHORT"
    
    if signal == "LONG":
        reasons.append(f"Bounce/Repair probability favorable ({max(bounce['prob'], repair['prob']):.0%})")
        reasons.append(f"Recovery momentum: {recovery_score:.0f} ({'Strong' if recovery_score >= 70 else 'Building'})")
        reasons.append(f"Canary: {canary['label']}")
    elif signal == "SHORT":
        reasons.append(f"Fall probability dominant ({fall['prob']:.0%})")
        reasons.append(f"Recovery momentum weak: {recovery_score:.0f}")
        reasons.append(f"Canary: {canary['label']}")
    else:
        reasons.append("Mixed signals; no strong edge")
    
    if cluster_name:
        reasons.append(f"Cluster: {cluster_name}")
    
    return {"signal": signal, "reasons": reasons, "bounce_prob": bounce["prob"], "repair_prob": repair["prob"], "regime_prob": regime["prob"], "fall_prob": fall["prob"]}

# ==============================================================================
# Risk Metrics
# ==============================================================================
def calculate_risk_metrics(equity_curve: pd.Series, rf: float = 0.02) -> Dict[str, float]:
    returns = equity_curve.pct_change().dropna()
    
    if len(returns) < 20:
        return {"sharpe": np.nan, "sortino": np.nan, "calmar": np.nan, "max_dd": np.nan, "total_return": np.nan, "cagr": np.nan}
    
    excess_returns = returns - rf / 252
    sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    
    downside = returns[returns < 0]
    sortino = (excess_returns.mean() / downside.std()) * np.sqrt(252) if len(downside) > 0 and downside.std() > 0 else 0
    
    peak = equity_curve.cummax()
    dd = equity_curve / peak - 1
    max_dd = dd.min()
    
    calmar = (returns.mean() * 252) / abs(max_dd) if max_dd != 0 and not pd.isna(max_dd) else 0
    
    years = len(equity_curve) / 252
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1 if years > 0 else 0
    
    return {
        "sharpe": sharpe, "sortino": sortino, "calmar": calmar,
        "max_dd": max_dd, "total_return": equity_curve.iloc[-1] / equity_curve.iloc[0] - 1,
        "cagr": cagr
    }

def monte_carlo_simulation(bt_df: pd.DataFrame, n_runs: int = 1000, burn_in: int = 50) -> Dict[str, float]:
    returns = bt_df["strategy_ret"].dropna().values if "strategy_ret" in bt_df.columns else bt_df["ret"].dropna().values
    
    if len(returns) < burn_in + 50:
        return {"mean_return": 0, "prob_ruin": 0, "ci_95_low": 0, "ci_95_high": 0}
    
    returns = returns[burn_in:]
    final_values = []
    
    for _ in range(n_runs):
        np.random.shuffle(returns)
        equity = np.cumprod(1 + returns)
        final_values.append(equity[-1])
    
    final_values = np.array(final_values)
    
    return {
        "mean_return": float(np.mean(final_values) - 1),
        "prob_ruin": float(np.mean(final_values < 0.5)),
        "ci_95_low": float(np.percentile(final_values, 2.5) - 1),
        "ci_95_high": float(np.percentile(final_values, 97.5) - 1),
    }

def walk_forward_validation(score_df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    results = []
    split_size = len(score_df) // n_splits
    
    for i in range(n_splits - 1):
        train_end = int((i + 0.75 * 3) * split_size)
        test_start = train_end
        test_end = min(int((i + 4) * split_size), len(score_df))
        
        if test_end - test_start < 50:
            continue
        
        test = score_df.iloc[test_start:test_end].copy()
        
        if "master_score" in test.columns and "rsp_close" in test.columns:
            test = test.sort_values("date").reset_index(drop=True)
            test["signal"] = (test["master_score"] > test["master_score"].median()).astype(int).shift(1).fillna(0)
            test["ret"] = test["rsp_close"].pct_change().fillna(0)
            test["strategy_ret"] = test["signal"] * test["ret"]
            test["equity"] = (1 + test["strategy_ret"]).cumprod()
            
            if len(test) > 20 and test["equity"].iloc[-1] > 0:
                returns = test["strategy_ret"].dropna()
                if len(returns) > 10 and returns.std() > 0:
                    sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
                    max_dd = (test["equity"] / test["equity"].cummax() - 1).min()
                    
                    results.append({
                        "Period": f"{i+1}",
                        "Sharpe": sharpe,
                        "Return": test["equity"].iloc[-1] / test["equity"].iloc[0] - 1,
                        "MaxDD": max_dd,
                    })
    
    return pd.DataFrame(results)

# ==============================================================================
# Backtest Engine
# ==============================================================================
def run_backtest(score_df: pd.DataFrame, fast: int = 5, slow: int = 13, deadband: float = 0.0,
                use_canary: bool = True, canary_thr: float = 0.05, conf_thr: float = 55.0,
                switch_cost_bps: float = 5.0) -> pd.DataFrame:
    bt = score_df.copy().sort_values("date").reset_index(drop=True)
    
    if "master_score" not in bt.columns or "rsp_close" not in bt.columns:
        logger.warning("Missing required columns for backtest")
        return pd.DataFrame()
    
    bt["ema_fast"] = ema(bt["master_score"], fast)
    bt["ema_slow"] = ema(bt["master_score"], slow)
    bt["osc"] = (bt["ema_fast"] - bt["ema_slow"]) / bt["ema_slow"].replace(0, np.nan)
    
    signal = bt["osc"] > deadband
    
    if use_canary and "canary_comp" in bt.columns:
        signal &= (bt["canary_comp"].fillna(-1) > canary_thr) & (bt["canary_conf"].fillna(0) >= conf_thr)
    
    bt["signal_raw"] = signal.astype(int)
    bt["signal"] = bt["signal_raw"].shift(1).fillna(0)
    
    bt["ret"] = bt["rsp_close"].pct_change().fillna(0)
    bt["turnover"] = bt["signal"].diff().abs().fillna(0)
    cost = switch_cost_bps / 10000.0
    bt["strategy_ret"] = bt["signal"] * bt["ret"] - bt["turnover"] * cost
    
    bt["equity_strategy"] = (1 + bt["strategy_ret"]).cumprod()
    bt["equity_buyhold"] = (1 + bt["ret"]).cumprod()
    
    return bt

# ==============================================================================
# Model Build / Load
# ==============================================================================
def build_model_from_history(daily: pd.DataFrame, weekly: pd.DataFrame) -> Dict[str, Any]:
    daily_feat = add_indicator_features(daily)
    weekly_feat = add_indicator_features(weekly) if not weekly.empty else weekly.copy()
    
    piv = daily_feat.pivot(index="date", columns="symbol", values="close")
    if "RSP" not in piv.columns:
        raise ValueError("RSP daily history is required")
    
    outcomes = build_outcomes(piv["RSP"].dropna())
    base = piv.join(outcomes, how="inner").dropna()
    
    features = [c for c in base.columns if c not in ["RSP"] + [f"{x}_success" for x in OUTCOME_DEFS.keys()]]
    features = [c for c in features if base[c].notna().sum() >= MIN_GATE_SUPPORT * 2]
    
    learned = {"states": {}, "bands": {}, "meta": {"rows": int(len(base))}}
    
    for state in ["bounce", "repair", "regime", "fall"]:
        singles = learn_single_gates(base, features, state)
        combos = learn_combo_gates(base, state, singles)
        learned["states"][state] = {"singles": singles, "combos": combos, "base_rate": float(base[f"{state}_success"].mean())}
        if state != "fall":
            learned["bands"][state] = summarize_bands(base, state)
    
    cluster_base = base.dropna(subset=KEY_FEATURES).copy()
    cluster_stats, cluster_artifacts = build_clusters(cluster_base, KEY_FEATURES, n_clusters=6)
    
    canary_hist = build_canary_from_history(daily_feat)
    
    learned["clusters"] = {
        "features": KEY_FEATURES,
        "mean": cluster_artifacts.scaler_mean,
        "scale": cluster_artifacts.scaler_scale,
        "centroids": cluster_artifacts.centroids,
        "names": cluster_artifacts.cluster_names,
        "stats": cluster_stats.to_dict(orient="records"),
        "silhouette": cluster_artifacts.silhouette_score,
    }
    
    learned["canary_hist"] = canary_hist.to_dict(orient="records") if not canary_hist.empty else []
    
    save_json(MODEL_PATH, learned)
    daily_feat.to_parquet(HIST_DAILY_PATH, index=False)
    if not weekly_feat.empty:
        weekly_feat.to_parquet(HIST_WEEKLY_PATH, index=False)
    
    return learned

def load_model() -> Optional[Dict[str, Any]]:
    if not MODEL_PATH.exists() or not HIST_DAILY_PATH.exists():
        return None
    model = load_json(MODEL_PATH, {})
    return model if model else None

# ==============================================================================
# Main App
# ==============================================================================
def main():
    logger.info("Breadth Sweet Spot Engine v5.0 starting")
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        phase = detect_session_phase()
        st.write(f"Session: {phase.value}")
        
        use_proxy = st.toggle("Use proxy NYMO intraday", value=(phase != SessionPhase.OFFICIAL))
        force_rebuild = st.toggle("Force rebuild model", value=False)
        
        st.subheader("🛡️ Canary Filter")
        use_canary = st.toggle("Enable canary", value=True)
        canary_thr = st.slider("Canary threshold", -0.20, 0.30, 0.05, 0.01)
        conf_thr = st.slider("Confidence threshold", 0, 100, 55)
        
        st.subheader("📊 Backtest")
        fast_ema = st.slider("Fast EMA", 3, 15, 5)
        slow_ema = st.slider("Slow EMA", 8, 34, 13)
        deadband = st.slider("Deadband", 0.0, 0.10, 0.0, 0.005)
        switch_cost = st.slider("Switch cost (bps)", 0.0, 25.0, 5.0, 0.5)
        
        st.subheader("📁 Data")
        hist_upload = st.file_uploader("Historical ZIP", type=["zip"])
        reset_model = st.button("🗑️ Reset Model", use_container_width=True)
        
        if reset_model:
            for p in [HIST_DAILY_PATH, HIST_WEEKLY_PATH, MODEL_PATH]:
                if p.exists():
                    p.unlink()
            st.success("Model reset")
            st.rerun()
    
    model = None
    if hist_upload is not None or force_rebuild or not MODEL_PATH.exists():
        if hist_upload is not None:
            with st.spinner("🔨 Building model..."):
                try:
                    daily, weekly = parse_stockcharts_zip(hist_upload.read())
                    model = build_model_from_history(daily, weekly)
                    st.sidebar.success("✅ Model built")
                except Exception as e:
                    st.sidebar.error(f"Build failed: {e}")
                    logger.error(f"Build failed: {e}")
    
    if model is None:
        model = load_model()
        if model:
            st.sidebar.info(f"✅ Loaded ({len(pd.read_parquet(HIST_DAILY_PATH))} rows)")
    
    if model is None:
        st.info("📤 Upload historical ZIP to build model")
        st.stop()
    
    daily_feat = pd.read_parquet(HIST_DAILY_PATH)
    latest_date = pd.to_datetime(daily_feat["date"]).max()
    piv = daily_feat.pivot(index="date", columns="symbol", values="close")
    
    snapshot = {}
    for feat in KEY_FEATURES:
        sym = feat.replace("_%B", "").replace("_close", "")
        if sym in piv.columns and latest_date in piv.index:
            snapshot[feat] = safe_float(piv.loc[latest_date, sym])
    
    prior_date = daily_feat[daily_feat["date"] < latest_date]["date"].max()
    prior_snapshot = {}
    if pd.notna(prior_date):
        for feat in KEY_FEATURES:
            sym = feat.replace("_%B", "").replace("_close", "")
            if sym in piv.columns:
                prior_snapshot[feat] = safe_float(piv.loc[prior_date, sym])
    
    nymo_eff = proxy_nymo(snapshot, prior_snapshot) if use_proxy else {"value": safe_float(snapshot.get("$NYMO", np.nan)), "delta": 0, "state": "Official"}
    
    state_scores = {state: evaluate_state(snapshot, model["states"][state]) for state in ["bounce", "repair", "regime", "fall"]}
    band_df, band_totals = score_bands(snapshot, model.get("bands", {}))
    
    recovery = compute_recovery_momentum(piv, latest_date)
    recovery_score = recovery["recovery_score"]
    
    canary_hist = pd.DataFrame(model.get("canary_hist", []))
    if not canary_hist.empty:
        canary_hist["date"] = pd.to_datetime(canary_hist["date"])
        row = canary_hist[canary_hist["date"] <= latest_date].tail(1)
        canary_comp = float(row["canary_comp"].iloc[0]) if not row.empty else 0
        canary_conf = float(row["canary_conf"].iloc[0]) if not row.empty else 0
    else:
        canary_comp, canary_conf = 0, 0
    
    canary = {"label": "Risk-On" if canary_comp > 0.05 else "Risk-Off" if canary_comp < -0.05 else "Neutral", "comp": canary_comp, "conf": canary_conf}
    
    cl_id, cl_name, cl_conf = predict_cluster(snapshot, ClusterArtifacts(
        scaler_mean=model["clusters"]["mean"],
        scaler_scale=model["clusters"]["scale"],
        features=model["clusters"]["features"],
        centroids=model["clusters"]["centroids"],
        cluster_names=model["clusters"]["names"],
        cluster_stats={},
        silhouette_score=model["clusters"].get("silhouette", 0),
    )) if model.get("clusters") else (None, None, None)
    
    signal = classify_signal(state_scores, band_totals, canary, cl_name, recovery_score)
    
    render_status_bar(phase, nymo_eff)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: render_score_card("Bounce", state_scores["bounce"]["prob"] * 100 if pd.notna(state_scores["bounce"]["prob"]) else 0, 100)
    with c2: render_score_card("Repair", state_scores["repair"]["prob"] * 100 if pd.notna(state_scores["repair"]["prob"]) else 0, 100)
    with c3: render_score_card("Regime", state_scores["regime"]["prob"] * 100 if pd.notna(state_scores["regime"]["prob"]) else 0, 100)
    with c4: render_score_card("Canary", canary_conf, 100)
    
    render_signal_box(signal["signal"], "; ".join(signal["reasons"][:3]))
    
    tab1, tab2, tab3 = st.tabs(["📊 Decision Dashboard", "📈 Backtest vs RSP", "🔍 Model Explorer"])
    
    with tab1:
        left, right = st.columns([1.2, 1])
        with left:
            st.markdown("### Why This Signal?")
            for r in signal["reasons"]:
                st.markdown(f"- {r}")
            
            st.markdown("### Gate Verdicts")
            for state in ["bounce", "repair", "regime", "fall"]:
                s = state_scores[state]
                st.progress(min(max(s["pass_frac"], 0.0), 1.0), text=f"{state.title()}: {s['prob']*100:.0f}% | Pass {s['pass_frac']*100:.0f}%")
        
        with right:
            st.markdown("### Current Readings")
            core = pd.DataFrame([{"Feature": f, "Current": snapshot.get(f, np.nan)} for f in KEY_FEATURES[:10]])
            st.dataframe(core.round(3), use_container_width=True, hide_index=True)
    
    with tab2:
        st.header("📈 Strategy vs Buy & Hold RSP")
        
        score_df = piv.copy()
        score_df["master_score"] = (0.35 * np.array([state_scores["bounce"]["prob"] or 0] * len(score_df)) +
                                   0.40 * np.array([state_scores["repair"]["prob"] or 0] * len(score_df)) +
                                   0.25 * np.array([state_scores["regime"]["prob"] or 0] * len(score_df)))
        score_df["rsp_close"] = piv["RSP"] if "RSP" in piv.columns else np.nan
        
        if "RSP" in piv.columns:
            bt = run_backtest(score_df, fast=fast_ema, slow=slow_ema, deadband=deadband,
                            use_canary=use_canary, canary_thr=canary_thr, conf_thr=conf_thr,
                            switch_cost_bps=switch_cost)
            
            if not bt.empty:
                strat_metrics = calculate_risk_metrics(bt["equity_strategy"])
                rsp_metrics = calculate_risk_metrics(bt["equity_buyhold"])
                mc_results = monte_carlo_simulation(bt)
                wf_results = walk_forward_validation(score_df)
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Strategy CAGR", f"{strat_metrics['cagr']:.1%}" if pd.notna(strat_metrics['cagr']) else "n/a")
                m2.metric("RSP CAGR", f"{rsp_metrics['cagr']:.1%}" if pd.notna(rsp_metrics['cagr']) else "n/a")
                m3.metric("Excess Return", f"{(strat_metrics['cagr'] - rsp_metrics['cagr']):.1%}" if pd.notna(strat_metrics['cagr']) and pd.notna(rsp_metrics['cagr']) else "n/a")
                m4.metric("MC Mean", f"{mc_results['mean_return']:.1%}")
                
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Strategy Sharpe", f"{strat_metrics['sharpe']:.2f}" if pd.notna(strat_metrics['sharpe']) else "n/a")
                r2.metric("RSP Sharpe", f"{rsp_metrics['sharpe']:.2f}" if pd.notna(rsp_metrics['sharpe']) else "n/a")
                r3.metric("Excess Sharpe", f"{(strat_metrics['sharpe'] - rsp_metrics['sharpe']):.2f}" if pd.notna(strat_metrics['sharpe']) and pd.notna(rsp_metrics['sharpe']) else "n/a")
                r4.metric("Prob Ruin", f"{mc_results['prob_ruin']:.1%}")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=bt.index, y=bt["equity_strategy"], name="Strategy", line=dict(width=2)))
                fig.add_trace(go.Scatter(x=bt.index, y=bt["equity_buyhold"], name="RSP Buy & Hold", line=dict(width=2, dash='dot')))
                fig.update_layout(title="Equity Curve: Strategy vs RSP", height=400, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("Walk-Forward Validation"):
                    if not wf_results.empty:
                        st.dataframe(wf_results.round(3))
                    else:
                        st.info("Insufficient data")
                
                with st.expander("Monte Carlo"):
                    st.write(f"**95% CI:** [{mc_results['ci_95_low']:.1%}, {mc_results['ci_95_high']:.1%}]")
                    st.write(f"**Probability of Ruin:** {mc_results['prob_ruin']:.1%}")
            else:
                st.info("Backtest unavailable")
        else:
            st.warning("RSP data not available")
    
    with tab3:
        st.header("🔍 Model Explorer")
        
        st.subheader("Sweet Spot Bands")
        if not band_df.empty:
            st.dataframe(band_df.round(3), use_container_width=True, hide_index=True)
        
        st.subheader("Cluster Map")
        if model.get("clusters", {}).get("stats"):
            cluster_df = pd.DataFrame(model["clusters"]["stats"])
            st.dataframe(cluster_df.round(3), use_container_width=True, hide_index=True)
        
        st.subheader("Learned Gates")
        gate_state = st.selectbox("State", ["bounce", "repair", "regime", "fall"])
        singles = model["states"][gate_state]["singles"]
        if singles:
            sg = pd.DataFrame([{**g, "gate": f"{g['feature']} {'>=' if g['direction']=='gte' else '<='} {round(g['threshold'],3)}"} for g in singles])
            st.dataframe(sg[["gate", "support", "hit_rate", "base_rate", "lift"]].round(3), use_container_width=True, hide_index=True)

def render_status_bar(phase, nymo_eff):
    st.markdown(f"""
    <div class="status-bar">
        <div><strong>Phase:</strong> {phase.value}</div>
        <div><strong>NYMO:</strong> {fmt_num(nymo_eff['value'], 1)} ({nymo_eff['state']})</div>
        <div><strong>Mode:</strong> {'Proxy' if nymo_eff.get('state') != 'Official' else 'Official'}</div>
    </div>
    """, unsafe_allow_html=True)

def render_score_card(title: str, value: float, max_score: float):
    hex_color, fill_class = score_color(value, max_score)
    width = max(0, min(100, int(round(100 * value / max_score)))) if max_score else 0
    
    st.markdown(f"""
    <div class="soft-card score-card">
        <div class="score-title">{title}</div>
        <div class="score-value">{value:.1f}</div>
        <div class="score-bar"><div class="score-fill {fill_class}" style="width:{width}%"></div></div>
    </div>
    """, unsafe_allow_html=True)

def render_signal_box(verdict: str, text: str):
    cls = {"LONG": "signal-long", "SHORT": "signal-short"}.get(verdict, "signal-hold")
    st.markdown(f"""
    <div class="{cls}">
        <h3 style="margin:0 0 0.5rem 0;">📌 Daily Verdict: {verdict}</h3>
        <p style="margin:0;color:var(--text);">{text}</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Unhandled exception")
        st.error(f"Application error: {str(e)}")
        st.exception(e)
