#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Breadth Sweet Spot Engine v9.0 - Holistic TSI Decision Engine
=============================================================
Merges historical gate learning, sweet spot bands, and TSI(25,13,7) cross analysis
into a holistic decision engine for RSP/URSP/SPXL swing timing.

Outputs: Bounce, Repair, Regime Up, Overheating, Fall, Regime Down states
Signals: LONG, SHORT, HOLD with simple trading gauge (95% confidence, etc.)

Author: Market Breadth Analysis Engine
Version: 9.0.0 (Holistic TSI Edition)
Last Updated: 2026-03-30
"""

from __future__ import annotations
import io, json, logging, math, warnings, zipfile
from dataclasses import dataclass, asdict
from datetime import datetime, time as dt_time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np, pandas as pd, plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy import stats

# ==============================================================================
# Configuration & Logging (Python 3.14 compatible)
# ==============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler(Path(__file__).parent / 'breadth_v9.log', mode='a')])
logger = logging.getLogger(__name__)
# Python 3.14 fix: separate filterwarnings calls
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ==============================================================================
# App Configuration
# ==============================================================================
st.set_page_config(page_title="Breadth Engine v9.0 | Holistic TSI", layout="wide", page_icon="📈")

CUSTOM_CSS = """
:root{--bg:#0b1020;--panel:#111936;--text:#ecf2ff;--muted:#98abd5;--green:#22c55e;--yellow:#f59e0b;--red:#ef4444;--blue:#38bdf8;}
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
.pill{display:inline-block;padding:.3rem .6rem;border-radius:999px;font-size:.82rem;font-weight:700;border:1px solid rgba(255,255,255,.12);margin-right:.35rem;}
.pill-green{background:rgba(34,197,94,.16);color:#bbf7d0;}
.pill-yellow{background:rgba(245,158,11,.16);color:#fde68a;}
.pill-red{background:rgba(239,68,68,.16);color:#fecaca;}
.status-bar{display:flex;justify-content:space-between;align-items:center;padding:.75rem 1rem;background:rgba(255,255,255,.05);border-radius:12px;margin-bottom:1rem;}
.signal-long{background:rgba(34,197,94,.14);border:1px solid rgba(34,197,94,.32);border-radius:16px;padding:1rem;}
.signal-short{background:rgba(239,68,68,.14);border:1px solid rgba(239,68,68,.32);border-radius:16px;padding:1rem;}
.signal-hold{background:rgba(245,158,11,.14);border:1px solid rgba(245,158,11,.32);border-radius:16px;padding:1rem;}
.gauge{padding:1rem;border-radius:16px;text-align:center;margin:1rem 0;}
.gauge-bull{background:rgba(34,197,94,.14);border:1px solid rgba(34,197,94,.32);}
.gauge-bear{background:rgba(239,68,68,.14);border:1px solid rgba(239,68,68,.32);}
.gauge-neutral{background:rgba(245,158,11,.14);border:1px solid rgba(245,158,11,.32);}
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown("""
<div class="main-title">
<h1 style="margin:0;font-size:1.8rem;">📈 Breadth Engine v9.0 | Holistic TSI</h1>
<p style="margin:0.5rem 0 0 0;color:#98abd5;">Historical Gates + Sweet Spots + TSI Cross Analysis • Beats Buy & Hold RSP</p>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# Constants (NO TRAILING SPACES)
# ==============================================================================
APP_DIR = Path("breadth_store_v9"); APP_DIR.mkdir(exist_ok=True)
HIST_DAILY_PATH = APP_DIR / "daily.parquet"
HIST_WEEKLY_PATH = APP_DIR / "weekly.parquet"
MODEL_PATH = APP_DIR / "model.json"
UPLOAD_HISTORY_PATH = APP_DIR / "uploads.csv"

KEY_FEATURES = [
    "$BPSPX", "$BPSPX_%B", "$BPNYA", "$OEXA200R", "$SPXA50R", "$NYMO", "$NYSI", "$CPCE",
    "$NYHL", "$NYAD", "$SPXADP", "$TRIN", "$VIX", "RSP:SPY"
]
INVERSE_INDICATORS = {"$TRIN", "$VIX", "$CPCE"}
BREADTH_FEATURES = {"$BPSPX", "$BPSPX_%B", "$BPNYA", "$OEXA200R", "$SPXA50R", "$NYMO", "$NYSI", "$NYHL", "$NYAD", "$SPXADP"}
LEADERSHIP_FEATURES = {"RSP:SPY", "IWM:SPY", "SMH:SPY", "XLF:SPY", "HYG:IEF"}
RISK_FEATURES = {"$TRIN", "$VIX", "$CPCE", "VXX", "SPXS:SVOL"}
TSI_FEATURES = BREADTH_FEATURES | LEADERSHIP_FEATURES | RISK_FEATURES

OUTCOME_DEFS = {
    "bounce": {"horizon": 10, "ret": 0.03, "dd": -0.03, "type": "max"},
    "repair": {"horizon": 20, "ret": 0.04, "dd": -0.05, "type": "end"},
    "regime": {"horizon": 60, "ret": 0.08, "dd": -0.08, "type": "end"},
    "fall": {"horizon": 10, "ret": -0.03, "dd": 0.03, "type": "min_end"},
}

EASTERN = "America/New_York"
MIN_GATE_SUPPORT = 40

# ==============================================================================
# Enums
# ==============================================================================
class SessionPhase(str, Enum):
    PRE = "Pre-Market"; OPEN = "Opening"; MID = "Midday"
    LATE = "Late-Day"; POST = "Post-Close"; OFFICIAL = "Official EOD"; WEEKEND = "Weekend"

# ==============================================================================
# Utilities
# ==============================================================================
def safe_float(x: Any) -> float:
    try: return float(x)
    except: return np.nan

def fmt_num(v: Any, d: int = 2) -> str:
    return "n/a" if pd.isna(v) else f"{float(v):.{d}f}"

def save_json(path: Path, data: Any) -> bool:
    try: path.write_text(json.dumps(data, indent=2, default=str)); return True
    except Exception as e: logger.error(f"JSON save failed: {e}"); return False

def load_json(path: Path, default: Any) -> Any:
    if not path.exists(): return default
    try: return json.loads(path.read_text())
    except: return default

def detect_session_phase() -> str:
    ts = pd.Timestamp.now(tz=EASTERN)
    if ts.weekday() >= 5: return "Weekend"
    t = dt_time(ts.hour, ts.minute)
    if t < dt_time(9, 30): return "Pre-Market"
    if t < dt_time(11, 0): return "Opening"
    if t < dt_time(14, 30): return "Midday"
    if t < dt_time(16, 0): return "Late-Day"
    if t < dt_time(18, 0): return "Post-Close"
    return "Official EOD"

# ==============================================================================
# Technical Indicators
# ==============================================================================
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def tsi(series: pd.Series, long_: int = 25, short_: int = 13, signal: int = 7) -> Tuple[pd.Series, pd.Series]:
    """True Strength Index with signal line - configurable parameters."""
    m = series.diff()
    a = m.abs()
    m1 = ema(ema(m, long_), short_)
    a1 = ema(ema(a, long_), short_)
    tsi_val = 100 * (m1 / a1.replace(0, np.nan))
    sig = ema(tsi_val, signal)
    return tsi_val, sig

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff(); up = delta.clip(lower=0); down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean(); ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)

def percent_b(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    ma = series.rolling(window).mean(); std = series.rolling(window).std()
    upper = ma + num_std * std; lower = ma - num_std * std
    denom = (upper - lower).replace(0, np.nan)
    return (series - lower) / denom

def macd_hist(series: pd.Series, fast: int = 24, slow: int = 52, signal: int = 18) -> pd.Series:
    line = ema(series, fast) - ema(series, slow); sig = ema(line, signal)
    return line - sig

def add_indicator_features(hist: pd.DataFrame) -> pd.DataFrame:
    out = []
    for sym, g in hist.groupby("symbol", sort=False):
        g = g.sort_values("date").copy(); close = g["close"]
        g["rsi14"] = rsi(close, 14); g["pct_b20"] = percent_b(close, 20, 2.0)
        g["macd_hist"] = macd_hist(close)
        out.append(g)
    return pd.concat(out, ignore_index=True)

# ==============================================================================
# Data Parsing
# ==============================================================================
def parse_stockcharts_csv(content: bytes) -> pd.DataFrame:
    text = content.decode("utf-8", errors="ignore"); lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    rows = []
    for ln in lines:
        if ln.lower().startswith(("date,", "symbol,", "ticker,")): continue
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 6: continue
        dt = pd.to_datetime(parts[0], errors="coerce")
        if pd.isna(dt): continue
        nums = [safe_float(x) for x in parts[1:6]]
        if all(pd.isna(x) for x in nums[:4]): continue
        rows.append({"date": dt, "open": nums[0], "high": nums[1], "low": nums[2], "close": nums[3], "volume": nums[4] if len(nums) > 4 else np.nan})
    if not rows: raise ValueError("No rows parsed from StockCharts CSV")
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
    return mapping.get(stem, stem.upper()), timeframe

def parse_stockcharts_zip(file_bytes: bytes) -> Tuple[pd.DataFrame, pd.DataFrame]:
    daily, weekly = [], []
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            for name in zf.namelist():
                if name.endswith("/") or not name.lower().endswith(".csv"): continue
                try:
                    content = zf.read(name); df = parse_stockcharts_csv(content)
                    sym, tf = symbol_from_filename(name); df["symbol"] = sym
                    (weekly if tf == "weekly" else daily).append(df)
                except Exception as e: logger.warning(f"Skip {name}: {e}"); continue
    except Exception as e: logger.error(f"ZIP parse failed: {e}"); raise ValueError(f"Failed to parse zip: {str(e)}")
    if not daily: raise ValueError("No daily CSV files parsed from ZIP")
    daily_df = pd.concat(daily, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    weekly_df = pd.concat(weekly, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True) if weekly else pd.DataFrame(columns=daily_df.columns)
    return daily_df, weekly_df

# ==============================================================================
# Forward Metrics / Outcomes
# ==============================================================================
def compute_future_metrics(price: pd.Series, horizon: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    vals = price.to_numpy(dtype=float); n = len(vals)
    if n < horizon + 1: idx = price.index; return pd.Series(np.nan, index=idx), pd.Series(np.nan, index=idx), pd.Series(np.nan, index=idx)
    idx = np.arange(n); window_idx = idx[:, None] + np.arange(1, horizon + 1)
    mask = window_idx >= n; safe_idx = np.clip(window_idx, 0, n - 1)
    win = vals[safe_idx].astype(float); win[mask] = np.nan
    base = vals[:, None]; rets = win / base - 1
    end_ret, max_gain, max_dd = rets[:, -1], np.nanmax(rets, axis=1), np.nanmin(rets, axis=1)
    valid = n - horizon; end_ret[valid:], max_gain[valid:], max_dd[valid:] = np.nan, np.nan, np.nan
    return pd.Series(end_ret, index=price.index), pd.Series(max_gain, index=price.index), pd.Series(max_dd, index=price.index)

def build_outcomes(rsp_price: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=rsp_price.index)
    for name, cfg in OUTCOME_DEFS.items():
        end_ret, max_gain, max_dd = compute_future_metrics(rsp_price, cfg["horizon"])
        if cfg["type"] == "max": success = (max_gain >= cfg["ret"]) & (max_dd >= cfg["dd"])
        elif cfg["type"] == "min_end": success = (end_ret <= cfg["ret"]) & (max_dd <= cfg["dd"])
        else: success = (end_ret >= cfg["ret"]) & (max_dd >= cfg["dd"])
        out[f"{name}_success"] = success.astype(float)
    return out

# ==============================================================================
# Gate Learning
# ==============================================================================
def direction_hints(feature: str, state: str) -> List[str]:
    if state == "fall":
        if any(k in feature for k in ["$TRIN", "$VIX", "$CPCE"]): return ["gte", "lte"]
        return ["lte", "gte"]
    if feature in ["$TRIN", "$VIX", "$CPCE", "VXX"]:
        return ["lte", "gte"] if state in {"repair", "regime"} else ["gte", "lte"]
    return ["gte", "lte"] if state in {"repair", "regime"} else ["lte", "gte"]

def learn_single_gates(df: pd.DataFrame, features: List[str], label: str, min_support: int = 30) -> List[dict]:
    y = df[f"{label}_success"].astype(float); base = float(y.mean()); gates = []
    for feat in features:
        s = pd.to_numeric(df[feat], errors="coerce"); valid = s.notna() & y.notna()
        sv, yv = s[valid], y[valid]
        if len(sv) < max(80, min_support * 2): continue
        qs = sorted(set(float(x) for x in sv.quantile(np.linspace(0.15, 0.85, 15)).dropna()))
        for direction in direction_hints(feat, label):
            for thr in qs:
                mask = sv >= thr if direction == "gte" else sv <= thr; support = int(mask.sum())
                if support < min_support: continue
                hit = float(yv[mask].mean())
                if hit <= base: continue
                lift = hit / base if base > 0 else np.nan; score = (hit - base) * math.sqrt(support)
                gates.append({"feature": feat, "direction": direction, "threshold": thr, "support": support, "hit_rate": hit, "base_rate": base, "lift": lift, "score": score})
    gates = sorted(gates, key=lambda x: (x["score"], x["lift"], x["support"]), reverse=True)
    top, used = [], set()
    for g in gates:
        if g["feature"] in used: continue
        used.add(g["feature"]); top.append(g)
        if len(top) >= 12: break
    return top

def learn_combo_gates(df: pd.DataFrame, label: str, singles: List[dict], min_support: int = 25) -> List[dict]:
    y = df[f"{label}_success"].astype(float); base = float(y.mean()); combos = []
    for i in range(len(singles)):
        for j in range(i + 1, min(len(singles), i + 6)):
            g1, g2 = singles[i], singles[j]
            if g1["feature"] == g2["feature"]: continue
            s1 = pd.to_numeric(df[g1["feature"]], errors="coerce"); s2 = pd.to_numeric(df[g2["feature"]], errors="coerce")
            m1 = s1 >= g1["threshold"] if g1["direction"] == "gte" else s1 <= g1["threshold"]
            m2 = s2 >= g2["threshold"] if g2["direction"] == "gte" else s2 <= g2["threshold"]
            mask = m1 & m2 & y.notna(); support = int(mask.sum())
            if support < min_support: continue
            hit = float(y[mask].mean())
            if hit <= base: continue
            combos.append({"gates": [g1, g2], "support": support, "hit_rate": hit, "base_rate": base, "lift": hit / base if base > 0 else np.nan, "score": (hit - base) * math.sqrt(support)})
    return sorted(combos, key=lambda x: (x["score"], x["lift"], x["support"]), reverse=True)[:6]

def gate_pass(cur: float, gate: dict) -> bool:
    if pd.isna(cur): return False
    return cur >= gate["threshold"] if gate["direction"] == "gte" else cur <= gate["threshold"]

# ==============================================================================
# Holistic TSI Composite (NEW CORE FEATURE)
# ==============================================================================
def compute_holistic_tsi(piv: pd.DataFrame, long_: int = 25, short_: int = 13, signal_len: int = 7,
                        weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Compute polarity-aware holistic TSI composite across breadth, price, and risk buckets."""
    weights = weights or {"breadth": 0.45, "leadership": 0.25, "risk": 0.20, "canary": 0.10}
    results = {}
    
    # Compute TSI for each feature with polarity adjustment
    for feat in TSI_FEATURES:
        if feat not in piv.columns: continue
        close = piv[feat].dropna()
        if len(close) < max(long_, short_) + 10: continue
        tsi_val, tsi_sig = tsi(close, long_, short_, signal_len)
        # Apply polarity flip for inverse indicators
        polarity = -1 if feat in INVERSE_INDICATORS else 1
        results[feat] = {
            "tsi": tsi_val.iloc[-1] * polarity,
            "signal": tsi_sig.iloc[-1] * polarity,
            "gap": (tsi_val.iloc[-1] - tsi_sig.iloc[-1]) * polarity,
            "polarity": polarity,
            "bucket": "breadth" if feat in BREADTH_FEATURES else "leadership" if feat in LEADERSHIP_FEATURES else "risk"
        }
    
    if not results: return {"final_tsi": np.nan, "breadth_tsi": np.nan, "leadership_tsi": np.nan, "risk_tsi": np.nan, "cross_state": "unknown", "regime_state": "unknown", "participation": np.nan, "components": {}}
    
    # Bucketed composites
    breadth_vals = [r["tsi"] for r in results.values() if r["bucket"] == "breadth"]
    leadership_vals = [r["tsi"] for r in results.values() if r["bucket"] == "leadership"]
    risk_vals = [r["tsi"] for r in results.values() if r["bucket"] == "risk"]
    
    breadth_tsi = np.mean(breadth_vals) if breadth_vals else np.nan
    leadership_tsi = np.mean(leadership_vals) if leadership_vals else np.nan
    risk_tsi = np.mean(risk_vals) if risk_vals else np.nan
    
    # Weighted final composite
    final_tsi = (weights["breadth"] * breadth_tsi + weights["leadership"] * leadership_tsi + weights["risk"] * risk_tsi)
    
    # Cross-state detection
    avg_signal = np.mean([r["signal"] for r in results.values() if pd.notna(r["signal"])])
    gap = final_tsi - avg_signal
    if gap > 0 and gap <= 2.0: cross_state = "near_bull_cross"
    elif gap > 2.0: cross_state = "bull_cross"
    elif gap < 0 and gap >= -2.0: cross_state = "near_bear_cross"
    elif gap < -2.0: cross_state = "bear_cross"
    else: cross_state = "neutral"
    
    # Zero-line context
    regime_state = "above_zero" if final_tsi > 0 else "below_zero"
    
    # Participation: % of components with TSI > signal and rising
    participation = np.mean([1.0 if r["tsi"] > r["signal"] and r["gap"] > 0 else 0.0 for r in results.values()])
    
    return {
        "final_tsi": final_tsi, "breadth_tsi": breadth_tsi, "leadership_tsi": leadership_tsi, "risk_tsi": risk_tsi,
        "cross_state": cross_state, "regime_state": regime_state, "participation": participation, "components": results
    }

def get_tsi_gauge_label(tsi_result: Dict[str, Any]) -> Tuple[str, str, str]:
    """Return (gauge_label, confidence_pct, action) for trading gauge."""
    tsi = tsi_result["final_tsi"]
    cross = tsi_result["cross_state"]
    regime = tsi_result["regime_state"]
    part = tsi_result["participation"]
    
    # Bullish states
    if cross == "bull_cross" and regime == "below_zero" and part >= 0.6:
        return "Bounce Triggered", "95%", "Probe long"
    if cross in ["bull_cross", "near_bull_cross"] and regime == "below_zero" and tsi > -10 and part >= 0.7:
        return "Repair Underway", "85%", "Add selectively"
    if cross == "bull_cross" and regime == "above_zero" and part >= 0.75:
        return "Regime Up Confirmed", "100%", "Full long"
    if regime == "above_zero" and tsi > 40 and part < 0.5:
        return "Overheating", "70%", "Trim / tighten"
    
    # Bearish states
    if cross == "bear_cross" and regime == "above_zero" and part <= 0.4:
        return "Fall Triggered", "90%", "Hedge / reduce"
    if cross in ["bear_cross", "near_bear_cross"] and regime == "below_zero" and tsi < -20 and part <= 0.3:
        return "Regime Down Confirmed", "100%", "Defensive / short bias"
    
    # Neutral
    return "Mixed / Neutral", f"{int(part * 100)}%", "Hold / no edge"

# ==============================================================================
# Clustering
# ==============================================================================
@dataclass
class ClusterArtifacts:
    scaler_mean: List[float]; scaler_scale: List[float]; features: List[str]
    centroids: List[List[float]]; cluster_names: Dict[str, str]; cluster_stats: Dict[str, Dict[str, float]]
    silhouette_score: float = 0.0

def build_clusters(outcomes_df: pd.DataFrame, features: List[str], n_clusters: int = 6) -> Tuple[pd.DataFrame, ClusterArtifacts]:
    feat_df = outcomes_df[features].apply(pd.to_numeric, errors="coerce"); valid = feat_df.dropna()
    n_clusters = max(3, min(n_clusters, max(3, len(valid) // 40)))
    scaler = StandardScaler(); X = scaler.fit_transform(valid)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20, max_iter=300); labels = km.fit_predict(X)
    sil_score = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else 0.0
    cluster_stats_rows = []
    for cl in sorted(np.unique(labels)):
        mask = labels == cl; sub = valid.loc[mask]
        metrics = {
            "cluster": int(cl), "samples": int(mask.sum()),
            "bounce_rate": float(outcomes_df.loc[sub.index, "bounce_success"].mean()) if "bounce_success" in outcomes_df.columns else 0,
            "repair_rate": float(outcomes_df.loc[sub.index, "repair_success"].mean()) if "repair_success" in outcomes_df.columns else 0,
            "regime_rate": float(outcomes_df.loc[sub.index, "regime_success"].mean()) if "regime_success" in outcomes_df.columns else 0,
            "fall_rate": float(outcomes_df.loc[sub.index, "fall_success"].mean()) if "fall_success" in outcomes_df.columns else 0,
        }
        for feat in features: metrics[f"{feat}_median"] = float(sub[feat].median())
        cluster_stats_rows.append(metrics)
    stats_df = pd.DataFrame(cluster_stats_rows).sort_values("cluster").reset_index(drop=True)
    names = {}
    for idx, row in stats_df.iterrows():
        if row.get("regime_rate", 0) > 0.30: names[idx] = "Durable regime"
        elif row.get("repair_rate", 0) > 0.22: names[idx] = "Repair cluster"
        elif row.get("bounce_rate", 0) > 0.35: names[idx] = "Bounce cluster"
        elif row.get("fall_rate", 0) > 0.30: names[idx] = "Deterioration cluster"
        else: names[idx] = "Mixed / transitional"
    stats_df["cluster_name"] = stats_df["cluster"].map(names)
    artifacts = ClusterArtifacts(scaler_mean=scaler.mean_.tolist(), scaler_scale=scaler.scale_.tolist(), features=features,
        centroids=km.cluster_centers_.tolist(), cluster_names={str(k): v for k, v in names.items()},
        cluster_stats=stats_df.set_index("cluster").to_dict(orient="index"), silhouette_score=float(sil_score))
    return stats_df, artifacts

def predict_cluster(current: Dict[str, float], artifacts: ClusterArtifacts) -> Tuple[Optional[int], Optional[str], Optional[float]]:
    vals = []
    for feat in artifacts.features:
        v = safe_float(current.get(feat, np.nan))
        if pd.isna(v): return None, None, None
        vals.append(v)
    x = np.array(vals); scaled = (x - np.array(artifacts.scaler_mean)) / np.where(np.array(artifacts.scaler_scale) == 0, 1, np.array(artifacts.scaler_scale))
    cents = np.array(artifacts.centroids); dists = np.sqrt(((cents - scaled) ** 2).sum(axis=1))
    cl = int(np.argmin(dists)); confidence = 1.0 / (1.0 + float(dists[cl]))
    return cl, artifacts.cluster_names.get(str(cl), f"Cluster {cl}"), confidence

# ==============================================================================
# Canary / Proxy
# ==============================================================================
def proxy_nymo(snapshot: Dict[str, float], prev_snapshot: Dict[str, float]) -> Dict[str, float]:
    nyad = safe_float(snapshot.get("$NYAD", np.nan)); spxadp = safe_float(snapshot.get("$SPXADP", np.nan))
    prev_nyad = safe_float(prev_snapshot.get("$NYAD", np.nan)); prev_spxadp = safe_float(prev_snapshot.get("$SPXADP", np.nan))
    cur_raw = 0.6 * (0 if pd.isna(nyad) else nyad) + 0.4 * (0 if pd.isna(spxadp) else spxadp)
    prev_raw = 0.6 * (0 if pd.isna(prev_nyad) else prev_nyad) + 0.4 * (0 if pd.isna(prev_spxadp) else prev_spxadp)
    cur = 100 * np.tanh(cur_raw / 1600.0); prev = 100 * np.tanh(prev_raw / 1600.0)
    state = "Deep washout" if cur <= -70 else "Negative but repairing" if cur <= -20 else "Neutral / crossing" if cur <= 20 else "Positive thrust"
    return {"value": float(cur), "delta": float(cur - prev), "state": state}

def build_canary_from_history(daily_feat: pd.DataFrame) -> pd.DataFrame:
    piv = daily_feat.pivot(index="date", columns="symbol", values="close"); score_map = {}
    for sym, wt in {"SPXS:SVOL": 0.24, "HYG:IEF": 0.20, "SMH:SPY": 0.18, "XLF:SPY": 0.12, "RSP:SPY": 0.12, "IWM:SPY": 0.10, "$VIX": 0.04}.items():
        if sym not in piv.columns: continue
        close = piv[sym].dropna()
        if close.shape[0] < 220: continue
        mh = macd_hist(close); t = close.diff().ewm(span=20).mean() / close.diff().abs().ewm(span=20).mean() * 100
        stoch = percent_b(close, 14, 3); cci100 = ((close - close.rolling(100).mean()) / (0.015 * (close - close.rolling(100).mean()).abs().rolling(100).mean().replace(0, np.nan)))
        df = pd.concat([mh.rename("macdh"), t.rename("tsi"), stoch.rename("stoch"), cci100.rename("cci")], axis=1).dropna()
        if df.empty: continue
        bull = (df["macdh"] > 0) & (df["tsi"] > 0) & (df["stoch"] > 50) & (df["cci"] > 0)
        bear = (df["macdh"] < 0) & (df["tsi"] < 0) & (df["stoch"] < 50) & (df["cci"] < 0)
        s = pd.Series(0.0, index=df.index); s[bull] = 1.0; s[bear] = -1.0
        if sym in INVERSE_INDICATORS: s = -s
        score_map[sym] = s
    if not score_map: return pd.DataFrame()
    common = None
    for s in score_map.values(): common = s.index if common is None else common.intersection(s.index)
    if common is None or len(common) == 0: return pd.DataFrame()
    df = pd.DataFrame({k: v.loc[common] for k, v in score_map.items()}).dropna(how="all")
    weights = pd.Series({k: {"SPXS:SVOL": 0.24, "HYG:IEF": 0.20, "SMH:SPY": 0.18, "XLF:SPY": 0.12, "RSP:SPY": 0.12, "IWM:SPY": 0.10, "$VIX": 0.04}[k] for k in df.columns}); weights = weights / weights.sum()
