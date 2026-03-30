#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Breadth Quant Engine Ultimate v3
--------------------------------
Practical breadth engine for daily use:
- one-time historical StockCharts ZIP upload with persistence
- daily snapshot CSV updates
- empirical gate learning for bounce / repair / regime / fall
- improvement-pattern features across 1/2/3/5/10 bars
- proxy NYMO / proxy NYSI for intraday use
- canary filter from historical ratio series
- cluster regime context
- LONG / SHORT / HOLD verdict
- HOLD setup parameters now show DIRECTIONALLY-CORRECT entry triggers
- on-demand backtest vs RSP / SPY buy & hold

This version intentionally favors trader-readable outputs over maximal complexity.
"""

from __future__ import annotations

import io
import json
import math
import zipfile
from dataclasses import dataclass
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# =============================================================================
# App config
# =============================================================================
st.set_page_config(page_title="Breadth Quant Engine Ultimate v3", layout="wide", page_icon="📈")

APP_DIR = Path("breadth_quant_store_v3")
APP_DIR.mkdir(exist_ok=True)
HIST_DAILY_PATH = APP_DIR / "daily_history.parquet"
HIST_WEEKLY_PATH = APP_DIR / "weekly_history.parquet"
MODEL_PATH = APP_DIR / "learned_model.json"
SNAPSHOT_DIR = APP_DIR / "snapshots"
SNAPSHOT_DIR.mkdir(exist_ok=True)

CUSTOM_CSS = """
<style>
:root{--bg:#0b1020;--panel:#111936;--text:#ecf2ff;--muted:#98abd5;--green:#22c55e;--yellow:#f59e0b;--red:#ef4444;--blue:#38bdf8;}
.block-container{padding-top:1rem;padding-bottom:2rem;}
.main-title{padding:1rem 1.2rem;border-radius:16px;background:linear-gradient(135deg, rgba(56,189,248,.18), rgba(167,139,250,.18));border:1px solid rgba(148,163,184,.22);margin-bottom:1rem;}
.soft-card{background:linear-gradient(180deg, rgba(17,25,54,.96), rgba(10,17,38,.98));border:1px solid rgba(148,163,184,.24);border-radius:18px;padding:1rem;box-shadow:0 10px 35px rgba(0,0,0,.22);margin-bottom:1rem;}
.score-title{color:#bcd0ff;font-size:1.02rem;font-weight:800;}
.score-value{font-size:2.7rem;font-weight:950;color:#fff;margin:.35rem 0;}
.pill{display:inline-block;padding:.3rem .6rem;border-radius:999px;font-size:.82rem;font-weight:700;border:1px solid rgba(255,255,255,.12);margin-right:.35rem;}
.pill-green{background:rgba(34,197,94,.16);color:#bbf7d0;}
.pill-yellow{background:rgba(245,158,11,.16);color:#fde68a;}
.pill-red{background:rgba(239,68,68,.16);color:#fecaca;}
.signal-long{background:rgba(34,197,94,.14);border:1px solid rgba(34,197,94,.32);border-radius:16px;padding:1rem;}
.signal-short{background:rgba(239,68,68,.14);border:1px solid rgba(239,68,68,.32);border-radius:16px;padding:1rem;}
.signal-hold{background:rgba(245,158,11,.14);border:1px solid rgba(245,158,11,.32);border-radius:16px;padding:1rem;}
.small-muted{color:#93a4cc;font-size:.88rem;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown(
    """
<div class="main-title">
<h1 style="margin:0;font-size:1.7rem;">📈 Breadth Quant Engine Ultimate v3</h1>
<p style="margin:.45rem 0 0 0;color:#98abd5;">Hard historical gates + improvement patterns + cluster + canary + practical entry parameters</p>
</div>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# Constants
# =============================================================================
EASTERN = "America/New_York"
KEY_FEATURES = [
    "$BPSPX", "$BPSPX_%B", "$BPNYA", "$OEXA200R", "$SPXA50R", "$NYMO", "$NYSI",
    "$CPCE", "$NYHL", "$NYAD", "$SPXADP", "$TRIN", "$VIX", "RSP:SPY"
]
INVERSE_INDICATORS = {"$TRIN", "$VIX", "$CPCE"}
TREND_WINDOWS = [1, 2, 3, 5, 10]
OUTCOME_DEFS = {
    "bounce": {"horizon": 10, "ret": 0.03, "dd": -0.03, "type": "max"},
    "repair": {"horizon": 20, "ret": 0.04, "dd": -0.05, "type": "end"},
    "regime": {"horizon": 60, "ret": 0.08, "dd": -0.08, "type": "end"},
    "fall":   {"horizon": 10, "ret": -0.03, "dd": 0.03, "type": "min_end"},
}
CANARY_WEIGHTS = {
    "SPXS:SVOL": 0.24,
    "HYG:IEF": 0.20,
    "SMH:SPY": 0.18,
    "XLF:SPY": 0.12,
    "RSP:SPY": 0.12,
    "IWM:SPY": 0.10,
    "$VIX": 0.04,
}
PROXY_NYMO_NORM = 1600.0
MIN_GATE_SUPPORT = 40

# Features whose rising is bullish in entry setup logic
BULLISH_UP_FEATURES = {
    "$BPSPX", "$BPSPX_%B", "$BPNYA", "$OEXA200R", "$SPXA50R", "$NYMO", "$NYSI",
    "$NYHL", "$NYAD", "$SPXADP", "RSP:SPY"
}
# Features whose falling is bullish
BULLISH_DOWN_FEATURES = {"$TRIN", "$VIX", "$CPCE"}

# =============================================================================
# Helpers
# =============================================================================
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


def score_css_class(v: float) -> str:
    if pd.isna(v):
        return "pill-yellow"
    if v >= 65:
        return "pill-green"
    if v <= 35:
        return "pill-red"
    return "pill-yellow"


def detect_session_phase() -> str:
    ts = pd.Timestamp.now(tz=EASTERN)
    if ts.weekday() >= 5:
        return "Weekend"
    t = dt_time(ts.hour, ts.minute)
    if t < dt_time(9, 30):
        return "Pre-Market"
    if t < dt_time(11, 0):
        return "Opening Window"
    if t < dt_time(14, 30):
        return "Midday Window"
    if t < dt_time(16, 0):
        return "Late-Day Window"
    if t < dt_time(18, 0):
        return "Post-Close"
    return "Official EOD"

# =============================================================================
# Technical indicators
# =============================================================================
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1 / period, adjust=False).mean()
    ma_down = down.ewm(alpha=1 / period, adjust=False).mean()
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
    frames = []
    for sym, g in hist.groupby("symbol", sort=False):
        g = g.sort_values("date").copy()
        close = g["close"]
        g["rsi14"] = rsi(close, 14)
        g["pct_b20"] = percent_b(close, 20, 2.0)
        g["macd_hist"] = macd_hist(close)
        g["stoch"] = stoch_from_close(close, 14, 3)
        for w in TREND_WINDOWS:
            g[f"d{w}"] = close.diff(w)
            g[f"roc{w}"] = 100 * (close / close.shift(w) - 1)
        frames.append(g)
    return pd.concat(frames, ignore_index=True)

# =============================================================================
# Parsing
# =============================================================================
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
        rows.append({
            "date": dt,
            "open": nums[0],
            "high": nums[1],
            "low": nums[2],
            "close": nums[3],
            "volume": nums[4] if len(nums) > 4 else np.nan,
        })
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
    return mapping.get(stem, stem.upper()), timeframe


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
        raise ValueError("No daily CSV files parsed from ZIP")
    daily_df = pd.concat(daily, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    weekly_df = pd.concat(weekly, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True) if weekly else pd.DataFrame(columns=daily_df.columns)
    return daily_df, weekly_df


def parse_snapshot_csv(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    if "Symbol" not in df.columns:
        raise ValueError("Snapshot CSV must contain Symbol column")
    close_col = None
    for c in ["Close", "Last", "Price", "Current", "Value", "Daily Close"]:
        if c in df.columns:
            close_col = c
            break
    if close_col is None:
        raise ValueError("Snapshot CSV missing a close-like column")
    out = pd.DataFrame({
        "Symbol": df["Symbol"].astype(str).str.strip(),
        "Close": pd.to_numeric(df[close_col], errors="coerce")
    })
    return out.dropna(subset=["Close"])

# =============================================================================
# Forward metrics / outcomes
# =============================================================================
def compute_future_metrics(price: pd.Series, horizon: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    vals = price.to_numpy(dtype=float)
    n = len(vals)
    idx = price.index
    if n < horizon + 1:
        return pd.Series(np.nan, index=idx), pd.Series(np.nan, index=idx), pd.Series(np.nan, index=idx)

    window_idx = np.arange(n)[:, None] + np.arange(1, horizon + 1)
    mask = window_idx >= n
    safe_idx = np.clip(window_idx, 0, n - 1)
    win = vals[safe_idx].astype(float)
    win[mask] = np.nan
    base = vals[:, None]
    rets = win / base - 1

    end_ret = np.full(n, np.nan, dtype=float)
    max_gain = np.full(n, np.nan, dtype=float)
    max_dd = np.full(n, np.nan, dtype=float)

    valid_rows = ~np.all(np.isnan(rets), axis=1)
    if valid_rows.any():
        end_ret[valid_rows] = rets[valid_rows, -1]
        max_gain[valid_rows] = np.nanmax(rets[valid_rows], axis=1)
        max_dd[valid_rows] = np.nanmin(rets[valid_rows], axis=1)

    valid_cut = n - horizon
    end_ret[valid_cut:] = np.nan
    max_gain[valid_cut:] = np.nan
    max_dd[valid_cut:] = np.nan

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

# =============================================================================
# Gate learning
# =============================================================================
def direction_hints(feature: str, state: str) -> List[str]:
    if state == "fall":
        if feature in INVERSE_INDICATORS:
            return ["gte", "lte"]
        return ["lte", "gte"]
    if feature in INVERSE_INDICATORS:
        return ["lte", "gte"] if state in {"repair", "regime"} else ["gte", "lte"]
    return ["gte", "lte"] if state in {"repair", "regime"} else ["lte", "gte"]


def learn_single_gates(df: pd.DataFrame, features: List[str], label: str, min_support: int = MIN_GATE_SUPPORT) -> List[dict]:
    y = df[f"{label}_success"].astype(float)
    base = float(y.mean())
    gates: List[dict] = []
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
                gates.append({
                    "feature": feat, "direction": direction, "threshold": thr,
                    "support": support, "hit_rate": hit, "base_rate": base,
                    "lift": hit / base if base > 0 else np.nan, "score": score,
                })
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


def gate_pass(cur: float, gate: dict) -> bool:
    if pd.isna(cur):
        return False
    return cur >= gate["threshold"] if gate["direction"] == "gte" else cur <= gate["threshold"]


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
                "base_rate": base, "lift": hit / base if base > 0 else np.nan, "score": score,
            })
    combos = sorted(combos, key=lambda x: (x["score"], x["lift"], x["support"]), reverse=True)
    return combos[:6]

# =============================================================================
# Clustering
# =============================================================================
@dataclass
class ClusterArtifacts:
    scaler_mean: List[float]
    scaler_scale: List[float]
    features: List[str]
    centroids: List[List[float]]
    cluster_names: Dict[str, str]
    silhouette_score: float = 0.0


def assign_cluster_names(stats_df: pd.DataFrame) -> Dict[int, str]:
    names: Dict[int, str] = {}
    for idx, row in stats_df.iterrows():
        bounce = row.get("bounce_rate", 0)
        repair = row.get("repair_rate", 0)
        regime = row.get("regime_rate", 0)
        fall = row.get("fall_rate", 0)
        bb = row.get("$BPSPX_%B_median", np.nan)
        if fall >= max(repair, bounce, regime) and fall > 0.30:
            names[idx] = "Deterioration cluster"
        elif regime >= max(repair, bounce, fall) and regime > 0.30:
            names[idx] = "Durable regime"
        elif repair >= max(regime, bounce, fall) and repair > 0.22:
            names[idx] = "Repair cluster"
        elif bounce >= max(regime, repair, fall) and bounce > 0.35:
            names[idx] = "Bounce cluster"
        elif pd.notna(bb) and bb < 0.15:
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
    sil = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else 0.0
    rows = []
    for cl in sorted(np.unique(labels)):
        mask = labels == cl
        sub = valid.loc[mask]
        rows.append({
            "cluster": int(cl),
            "samples": int(mask.sum()),
            "bounce_rate": float(outcomes_df.loc[sub.index, "bounce_success"].mean()) if "bounce_success" in outcomes_df.columns else 0,
            "repair_rate": float(outcomes_df.loc[sub.index, "repair_success"].mean()) if "repair_success" in outcomes_df.columns else 0,
            "regime_rate": float(outcomes_df.loc[sub.index, "regime_success"].mean()) if "regime_success" in outcomes_df.columns else 0,
            "fall_rate": float(outcomes_df.loc[sub.index, "fall_success"].mean()) if "fall_success" in outcomes_df.columns else 0,
            **{f"{feat}_median": float(sub[feat].median()) for feat in features}
        })
    stats_df = pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)
    names = assign_cluster_names(stats_df)
    return stats_df, ClusterArtifacts(
        scaler_mean=scaler.mean_.tolist(),
        scaler_scale=scaler.scale_.tolist(),
        features=features,
        centroids=km.cluster_centers_.tolist(),
        cluster_names={str(k): v for k, v in names.items()},
        silhouette_score=float(sil),
    )


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

# =============================================================================
# Canary / proxy / scoring
# =============================================================================
def build_canary_from_history(daily_feat: pd.DataFrame) -> pd.DataFrame:
    piv = daily_feat.pivot(index="date", columns="symbol", values="close")
    scores = {}
    for sym in CANARY_WEIGHTS:
        if sym not in piv.columns:
            continue
        close = piv[sym].dropna()
        if close.shape[0] < 220:
            continue
        mh = macd_hist(close)
        tsi = close.diff().ewm(span=20).mean() / close.diff().abs().ewm(span=20).mean() * 100
        stoch = stoch_from_close(close, 14, 3)
        cci = (close - close.rolling(100).mean()) / (0.015 * (close - close.rolling(100).mean()).abs().rolling(100).mean().replace(0, np.nan))
        df = pd.concat([mh.rename("macdh"), tsi.rename("tsi"), stoch.rename("stoch"), cci.rename("cci")], axis=1).dropna()
        if df.empty:
            continue
        bull = (df["macdh"] > 0) & (df["tsi"] > 0) & (df["stoch"] > 50) & (df["cci"] > 0)
        bear = (df["macdh"] < 0) & (df["tsi"] < 0) & (df["stoch"] < 50) & (df["cci"] < 0)
        s = pd.Series(0.0, index=df.index)
        s[bull] = 1.0
        s[bear] = -1.0
        if sym in INVERSE_INDICATORS:
            s = -s
        scores[sym] = s
    if not scores:
        return pd.DataFrame()
    common = None
    for s in scores.values():
        common = s.index if common is None else common.intersection(s.index)
    if common is None or len(common) == 0:
        return pd.DataFrame()
    df = pd.DataFrame({k: v.loc[common] for k, v in scores.items()}).dropna(how="all")
    weights = pd.Series({k: CANARY_WEIGHTS[k] for k in df.columns})
    weights = weights / weights.sum()
    comp = (df * weights).sum(axis=1)
    sign = np.sign(comp.replace(0, np.nan))
    align = np.sign(df).replace(0, np.nan).eq(sign, axis=0).mean(axis=1).fillna(0)
    strength = df.abs().mean(axis=1).fillna(0)
    conf = ((0.6 * align) + (0.4 * strength)) * 100.0
    return pd.DataFrame({"canary_comp": comp, "canary_conf": conf})


def proxy_nymo(snapshot: Dict[str, float], prev_snapshot: Dict[str, float]) -> Dict[str, float]:
    nyad = safe_float(snapshot.get("$NYAD", np.nan))
    spxadp = safe_float(snapshot.get("$SPXADP", np.nan))
    prev_nyad = safe_float(prev_snapshot.get("$NYAD", np.nan))
    prev_spxadp = safe_float(prev_snapshot.get("$SPXADP", np.nan))
    cur_raw = 0.6 * (0 if pd.isna(nyad) else nyad) + 0.4 * (0 if pd.isna(spxadp) else spxadp)
    prev_raw = 0.6 * (0 if pd.isna(prev_nyad) else prev_nyad) + 0.4 * (0 if pd.isna(prev_spxadp) else prev_spxadp)
    cur = 100 * np.tanh(cur_raw / PROXY_NYMO_NORM)
    prev = 100 * np.tanh(prev_raw / PROXY_NYMO_NORM)
    state = "Deep washout" if cur <= -70 else "Negative but repairing" if cur <= -20 else "Neutral / crossing" if cur <= 20 else "Positive thrust"
    return {"value": float(cur), "delta": float(cur - prev), "state": state}


def compute_recovery_momentum(piv: pd.DataFrame, latest_date) -> Dict[str, Any]:
    features = ["$NYSI", "$NYMO", "$BPSPX", "$SPXA50R", "$NYHL", "$TRIN", "$CPCE", "$VIX", "RSP:SPY"]
    comps: Dict[str, float] = {}
    for feat in features:
        if feat not in piv.columns:
            comps[feat] = np.nan
            continue
        s = piv[feat].loc[:latest_date].dropna()
        if len(s) < 6:
            comps[feat] = np.nan
            continue
        y = s.iloc[-6:].to_numpy(dtype=float)
        x = np.arange(len(y))
        slope = np.polyfit(x, y, 1)[0]
        denom = max(np.nanmean(np.abs(y)), 1e-6)
        comps[feat] = slope / denom
    for feat in INVERSE_INDICATORS:
        if feat in comps and pd.notna(comps[feat]):
            comps[feat] = -comps[feat]
    vals = [np.tanh(v * 3.0) for v in comps.values() if pd.notna(v)]
    score = np.nan if not vals else max(0.0, min(100.0, float(np.mean(vals) * 50 + 50)))
    return {"recovery_score": score, "components": comps}


def summarize_bands(df: pd.DataFrame, label: str) -> Dict[str, Dict[str, float]]:
    hit = df[df[f"{label}_success"] == 1.0]
    res: Dict[str, Dict[str, float]] = {}
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
    rows = []
    totals = {}
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
    return {
        "prob": prob,
        "pass_frac": pass_frac,
        "base_rate": base,
        "passed_singles": [x for x in singles if x["passed"]],
        "all_singles": singles,
    }

# =============================================================================
# Model build / load
# =============================================================================
@st.cache_data(show_spinner=False)
def build_model_from_history_bytes(file_bytes: bytes) -> Dict[str, Any]:
    daily, weekly = parse_stockcharts_zip(file_bytes)
    daily_feat = add_indicator_features(daily)
    weekly_feat = add_indicator_features(weekly) if not weekly.empty else weekly.copy()

    piv = daily_feat.pivot(index="date", columns="symbol", values="close")
    if "RSP" not in piv.columns:
        raise ValueError("RSP daily history is required")

    outcomes = build_outcomes(piv["RSP"].dropna())
    base = piv.join(outcomes, how="inner").dropna()
    features = [c for c in base.columns if c not in ["RSP"] + [f"{x}_success" for x in OUTCOME_DEFS.keys()]]
    features = [c for c in features if base[c].notna().sum() >= MIN_GATE_SUPPORT * 2]

    learned: Dict[str, Any] = {"states": {}, "bands": {}, "meta": {"rows": int(len(base))}}
    for state in ["bounce", "repair", "regime", "fall"]:
        singles = learn_single_gates(base, features, state)
        combos = learn_combo_gates(base, state, singles)
        learned["states"][state] = {"singles": singles, "combos": combos, "base_rate": float(base[f"{state}_success"].mean())}
        if state != "fall":
            learned["bands"][state] = summarize_bands(base, state)

    cluster_base = base.dropna(subset=KEY_FEATURES).copy()
    cluster_stats, cluster_artifacts = build_clusters(cluster_base, KEY_FEATURES, n_clusters=6)
    learned["clusters"] = {
        "mean": cluster_artifacts.scaler_mean,
        "scale": cluster_artifacts.scaler_scale,
        "features": cluster_artifacts.features,
        "centroids": cluster_artifacts.centroids,
        "names": cluster_artifacts.cluster_names,
        "silhouette": cluster_artifacts.silhouette_score,
        "stats": cluster_stats.to_dict(orient="records"),
    }

    canary_hist = build_canary_from_history(daily_feat)
    learned["canary_hist"] = canary_hist.reset_index().rename(columns={"index": "date"}).to_dict(orient="records") if not canary_hist.empty else []

    daily_feat.to_parquet(HIST_DAILY_PATH, index=False)
    if not weekly_feat.empty:
        weekly_feat.to_parquet(HIST_WEEKLY_PATH, index=False)
    save_json(MODEL_PATH, learned)
    return learned


def load_model() -> Optional[Dict[str, Any]]:
    if not MODEL_PATH.exists():
        return None
    return load_json(MODEL_PATH, None)

# =============================================================================
# Snapshot construction
# =============================================================================
def build_snapshot_from_daily_and_upload(daily_feat: pd.DataFrame, snap_df: Optional[pd.DataFrame]) -> Tuple[Dict[str, float], Dict[str, float], pd.Timestamp, pd.Timestamp]:
    piv = daily_feat.pivot(index="date", columns="symbol", values="close")
    latest_date = pd.to_datetime(daily_feat["date"]).max()
    prior_date = pd.to_datetime(daily_feat[daily_feat["date"] < latest_date]["date"]).max()

    def extract_at(date_val: pd.Timestamp) -> Dict[str, float]:
        out = {}
        if pd.isna(date_val):
            return out
        for feat in KEY_FEATURES:
            sym = feat.replace("_%B", "")
            if feat.endswith("_%B"):
                if sym in piv.columns:
                    series = percent_b(piv[sym], 20, 2.0)
                    if date_val in series.index:
                        out[feat] = safe_float(series.loc[date_val])
            else:
                if sym in piv.columns and date_val in piv.index:
                    out[feat] = safe_float(piv.loc[date_val, sym])
        return out

    snapshot = extract_at(latest_date)
    prev_snapshot = extract_at(prior_date)

    if snap_df is not None and not snap_df.empty:
        rt_map = {str(r["Symbol"]).strip(): safe_float(r["Close"]) for _, r in snap_df.iterrows()}
        for sym, val in rt_map.items():
            snapshot[sym] = val
            if sym in piv.columns:
                tmp = piv[sym].copy()
                if latest_date in tmp.index:
                    tmp.loc[latest_date] = val
                snapshot[f"{sym}_%B"] = safe_float(percent_b(tmp, 20, 2.0).loc[latest_date]) if latest_date in tmp.index else np.nan
    return snapshot, prev_snapshot, latest_date, prior_date

# =============================================================================
# Verdict / HOLD setup logic
# =============================================================================
def classify_signal(state_scores: Dict[str, Any], canary: Dict[str, Any], cluster_name: Optional[str], recovery_score: float) -> Dict[str, Any]:
    bounce = state_scores["bounce"]
    repair = state_scores["repair"]
    regime = state_scores["regime"]
    fall = state_scores["fall"]

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

    signal = "HOLD"
    if long_flag:
        signal = "LONG"
    elif short_flag:
        signal = "SHORT"

    reasons = []
    if signal == "LONG":
        reasons.append(f"Bounce/Repair probability favorable ({max(bounce['prob'], repair['prob']):.0%})")
        reasons.append(f"Recovery momentum {recovery_score:.0f}")
        reasons.append(f"Canary: {canary['label']}")
    elif signal == "SHORT":
        reasons.append(f"Fall probability dominant ({fall['prob']:.0%})")
        reasons.append(f"Recovery momentum weak ({recovery_score:.0f})")
        reasons.append(f"Canary: {canary['label']}")
    else:
        reasons.append("Mixed signals; no strong edge")
        reasons.append(f"Canary: {canary['label']} | Recovery: {recovery_score:.0f}")
    if cluster_name:
        reasons.append(f"Cluster: {cluster_name}")

    return {
        "signal": signal,
        "reasons": reasons,
        "bounce_prob": bounce["prob"],
        "repair_prob": repair["prob"],
        "regime_prob": regime["prob"],
        "fall_prob": fall["prob"],
    }


def gate_operator_text(direction: str) -> str:
    return "≥" if direction == "gte" else "≤"


def gate_moves_toward_long(gate: dict) -> bool:
    feat = gate["feature"]
    if feat in BULLISH_UP_FEATURES:
        return gate["direction"] == "gte"
    if feat in BULLISH_DOWN_FEATURES:
        return gate["direction"] == "lte"
    return False


def gate_moves_toward_short(gate: dict) -> bool:
    feat = gate["feature"]
    if feat in BULLISH_UP_FEATURES:
        return gate["direction"] == "lte"
    if feat in BULLISH_DOWN_FEATURES:
        return gate["direction"] == "gte"
    return False


def gate_distance(snapshot: Dict[str, float], gate: dict) -> float:
    cur = safe_float(snapshot.get(gate["feature"], np.nan))
    thr = gate["threshold"]
    if pd.isna(cur) or pd.isna(thr):
        return np.inf
    if gate_pass(cur, gate):
        return 0.0
    denom = max(abs(thr), 1e-6)
    return abs(cur - thr) / denom


def build_hold_setup(snapshot: Dict[str, float], state_scores: Dict[str, Any]) -> Dict[str, List[dict]]:
    long_candidates: List[dict] = []
    short_candidates: List[dict] = []

    # Long setup should prioritize repair then bounce then regime confirmation
    for state_name in ["repair", "bounce", "regime"]:
        for g in state_scores[state_name].get("all_singles", []):
            if g["passed"]:
                continue
            if gate_moves_toward_long(g):
                long_candidates.append({**g, "from_state": state_name, "distance": gate_distance(snapshot, g)})

    # Short setup should prioritize fall then failed repair
    for state_name in ["fall"]:
        for g in state_scores[state_name].get("all_singles", []):
            if g["passed"]:
                continue
            if gate_moves_toward_short(g):
                short_candidates.append({**g, "from_state": state_name, "distance": gate_distance(snapshot, g)})

    # De-duplicate by feature, keep closest / strongest
    def dedupe_rank(cands: List[dict]) -> List[dict]:
        ranked = sorted(cands, key=lambda x: (x["distance"], -x["lift"], -x["support"]))
        out, used = [], set()
        for c in ranked:
            if c["feature"] in used:
                continue
            used.add(c["feature"])
            out.append(c)
            if len(out) >= 4:
                break
        return out

    return {"long": dedupe_rank(long_candidates), "short": dedupe_rank(short_candidates)}

# =============================================================================
# Backtest
# =============================================================================
def run_backtest(daily_feat: pd.DataFrame, use_spy: bool = True, fast: int = 5, slow: int = 13, deadband: float = 0.0, switch_cost_bps: float = 5.0) -> pd.DataFrame:
    piv = daily_feat.pivot(index="date", columns="symbol", values="close").sort_index()
    if "RSP" not in piv.columns:
        return pd.DataFrame()
    asset = "SPY" if use_spy and "SPY" in piv.columns else "RSP"
    close = piv[asset].dropna()
    if close.shape[0] < 100:
        return pd.DataFrame()
    score = (close / ema(close, fast)) - (close / ema(close, slow))
    sig = (score > deadband).astype(int).shift(1).fillna(0)
    ret = close.pct_change().fillna(0)
    turnover = sig.diff().abs().fillna(0)
    cost = switch_cost_bps / 10000.0
    strat = sig * ret - turnover * cost
    out = pd.DataFrame({
        "date": close.index,
        "close": close.values,
        "signal": sig.values,
        "ret": ret.values,
        "strategy_ret": strat.values,
    })
    out["equity_strategy"] = (1 + out["strategy_ret"]).cumprod()
    out["equity_buyhold"] = (1 + out["ret"]).cumprod()
    return out

# =============================================================================
# UI render helpers
# =============================================================================
def render_score_card(title: str, score: float):
    cls = score_css_class(score)
    st.markdown(
        f"""
<div class="soft-card">
  <div class="score-title">{title}</div>
  <div class="score-value">{0 if pd.isna(score) else int(round(score))}</div>
  <span class="pill {cls}">{fmt_num(score,1)}%</span>
</div>
""",
        unsafe_allow_html=True,
    )


def render_signal_box(signal: str, reason_text: str):
    klass = "signal-hold"
    if signal == "LONG":
        klass = "signal-long"
    elif signal == "SHORT":
        klass = "signal-short"
    st.markdown(f'<div class="{klass}"><div class="score-title">Daily Verdict</div><div class="score-value">{signal}</div><div>{reason_text}</div></div>', unsafe_allow_html=True)

# =============================================================================
# Main app
# =============================================================================
def main():
    with st.sidebar:
        st.header("⚙️ Configuration")
        phase = detect_session_phase()
        st.write(f"Session: {phase}")
        hist_upload = st.file_uploader("Historical ZIP", type=["zip"])
        snap_upload = st.file_uploader("Daily Snapshot CSV", type=["csv"])
        force_rebuild = st.toggle("Force rebuild model", value=False)
        run_bt = st.button("Run historical backtest")
        if st.button("Reset Model"):
            for p in [HIST_DAILY_PATH, HIST_WEEKLY_PATH, MODEL_PATH]:
                if p.exists():
                    p.unlink()
            st.success("Model reset. Reload the page.")
            st.stop()

    model = None
    if hist_upload is not None and (force_rebuild or not MODEL_PATH.exists()):
        with st.spinner("Building model from historical ZIP..."):
            model = build_model_from_history_bytes(hist_upload.read())
    else:
        model = load_model()

    if model is None:
        st.info("Upload historical ZIP once to build the model.")
        return

    daily_feat = pd.read_parquet(HIST_DAILY_PATH)
    snap_df = parse_snapshot_csv(snap_upload.read()) if snap_upload is not None else None
    snapshot, prev_snapshot, latest_date, prior_date = build_snapshot_from_daily_and_upload(daily_feat, snap_df)

    state_scores = {state: evaluate_state(snapshot, model["states"][state]) for state in ["bounce", "repair", "regime", "fall"]}
    band_df, band_totals = score_bands(snapshot, model.get("bands", {}))

    piv = daily_feat.pivot(index="date", columns="symbol", values="close").sort_index()
    recovery = compute_recovery_momentum(piv, latest_date)
    canary_hist = pd.DataFrame(model.get("canary_hist", []))
    canary = {"label": "Neutral", "comp": 0.0, "conf": 0.0}
    if not canary_hist.empty:
        canary_hist["date"] = pd.to_datetime(canary_hist["date"])
        row = canary_hist[canary_hist["date"] <= latest_date].tail(1)
        if not row.empty:
            comp = float(row["canary_comp"].iloc[0])
            conf = float(row["canary_conf"].iloc[0])
            canary = {
                "label": "Risk-On" if comp > 0.05 else "Risk-Off" if comp < -0.05 else "Neutral",
                "comp": comp,
                "conf": conf,
            }

    cluster_name, cluster_conf = None, None
    if model.get("clusters"):
        cl_id, cluster_name, cluster_conf = predict_cluster(snapshot, ClusterArtifacts(
            scaler_mean=model["clusters"]["mean"],
            scaler_scale=model["clusters"]["scale"],
            features=model["clusters"]["features"],
            centroids=model["clusters"]["centroids"],
            cluster_names=model["clusters"]["names"],
            silhouette_score=model["clusters"].get("silhouette", 0.0),
        ))

    nymo_eff = proxy_nymo(snapshot, prev_snapshot)
    verdict = classify_signal(state_scores, canary, cluster_name, recovery["recovery_score"] if pd.notna(recovery["recovery_score"]) else 0)
    hold_setup = build_hold_setup(snapshot, state_scores)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_score_card("Bounce", state_scores["bounce"]["prob"] * 100 if pd.notna(state_scores["bounce"]["prob"]) else np.nan)
    with c2:
        render_score_card("Repair", state_scores["repair"]["prob"] * 100 if pd.notna(state_scores["repair"]["prob"]) else np.nan)
    with c3:
        render_score_card("Regime", state_scores["regime"]["prob"] * 100 if pd.notna(state_scores["regime"]["prob"]) else np.nan)
    with c4:
        render_score_card("Fall", state_scores["fall"]["prob"] * 100 if pd.notna(state_scores["fall"]["prob"]) else np.nan)

    render_signal_box(verdict["signal"], " | ".join(verdict["reasons"][:3]))

    tab1, tab2, tab3 = st.tabs(["Decision Dashboard", "Backtest vs Buy & Hold", "Diagnostics"])

    with tab1:
        a, b = st.columns([1.2, 1])
        with a:
            st.markdown('<div class="soft-card">', unsafe_allow_html=True)
            st.markdown("**Why**")
            for r in verdict["reasons"]:
                st.markdown(f"- {r}")
            st.markdown(f"- Intraday Proxy NYMO: {fmt_num(nymo_eff['value'])} | Δ {fmt_num(nymo_eff['delta'])} | {nymo_eff['state']}")
            st.markdown(f"- Recovery Score: {fmt_num(recovery['recovery_score'],1)}")
            st.markdown(f"- Cluster Confidence: {fmt_num(cluster_conf,2) if cluster_conf is not None else 'n/a'}")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="soft-card">', unsafe_allow_html=True)
            st.markdown("**Current Snapshot**")
            cur_rows = [{"Feature": k, "Value": fmt_num(v, 3)} for k, v in snapshot.items() if k in KEY_FEATURES]
            st.dataframe(pd.DataFrame(cur_rows), width='stretch', hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with b:
            st.markdown('<div class="soft-card">', unsafe_allow_html=True)
            st.markdown("**Hold Trade Setup Parameters**")
            if verdict["signal"] == "HOLD":
                st.markdown("**Go LONG if these start to trigger:**")
                if hold_setup["long"]:
                    for g in hold_setup["long"]:
                        op = gate_operator_text(g["direction"])
                        st.markdown(
                            f"- `{g['feature']} {op} {fmt_num(g['threshold'],3)}` (now {fmt_num(g['current'],3)}) "
                            f"• state: {g['from_state']} • hit {fmt_num(g['hit_rate']*100,1)}%"
                        )
                else:
                    st.markdown("- No clean long confirmation gates available.")

                st.markdown("**Go SHORT if these start to trigger:**")
                if hold_setup["short"]:
                    for g in hold_setup["short"]:
                        op = gate_operator_text(g["direction"])
                        st.markdown(
                            f"- `{g['feature']} {op} {fmt_num(g['threshold'],3)}` (now {fmt_num(g['current'],3)}) "
                            f"• state: {g['from_state']} • hit {fmt_num(g['hit_rate']*100,1)}%"
                        )
                else:
                    st.markdown("- No clean short deterioration gates available.")
            else:
                st.markdown("Current verdict is not HOLD; setup parameters are less relevant.")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="soft-card">', unsafe_allow_html=True)
            st.markdown("**Gate Probabilities**")
            probs = pd.DataFrame([
                {"State": "Bounce", "Prob": fmt_num(verdict['bounce_prob'] * 100 if pd.notna(verdict['bounce_prob']) else np.nan, 1), "Pass Fraction": fmt_num(state_scores['bounce']['pass_frac'] * 100, 1)},
                {"State": "Repair", "Prob": fmt_num(verdict['repair_prob'] * 100 if pd.notna(verdict['repair_prob']) else np.nan, 1), "Pass Fraction": fmt_num(state_scores['repair']['pass_frac'] * 100, 1)},
                {"State": "Regime", "Prob": fmt_num(verdict['regime_prob'] * 100 if pd.notna(verdict['regime_prob']) else np.nan, 1), "Pass Fraction": fmt_num(state_scores['regime']['pass_frac'] * 100, 1)},
                {"State": "Fall", "Prob": fmt_num(verdict['fall_prob'] * 100 if pd.notna(verdict['fall_prob']) else np.nan, 1), "Pass Fraction": fmt_num(state_scores['fall']['pass_frac'] * 100, 1)},
            ])
            st.dataframe(probs, width='stretch', hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        if run_bt:
            with st.spinner("Running historical backtest..."):
                bt_rsp = run_backtest(daily_feat, use_spy=False)
                bt_spy = run_backtest(daily_feat, use_spy=True)
            if not bt_rsp.empty:
                st.markdown("**RSP Strategy vs RSP Buy & Hold**")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=bt_rsp["date"], y=bt_rsp["equity_strategy"], name="Strategy"))
                fig.add_trace(go.Scatter(x=bt_rsp["date"], y=bt_rsp["equity_buyhold"], name="RSP Buy & Hold"))
                if not bt_spy.empty:
                    fig.add_trace(go.Scatter(x=bt_spy["date"], y=bt_spy["equity_buyhold"], name="SPY Buy & Hold"))
                fig.update_layout(height=500, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, width='stretch')
            else:
                st.info("Backtest unavailable from current historical set.")
        else:
            st.info("Click **Run historical backtest** in the sidebar to compute strategy vs buy & hold.")

    with tab3:
        st.markdown("**Band Alignment**")
        st.dataframe(band_df, width='stretch', hide_index=True)
        st.markdown("**Passed Bounce/Repair/Regime/Fall Gates**")
        for state_name in ["bounce", "repair", "regime", "fall"]:
            st.markdown(f"**{state_name.title()}**")
            rows = []
            for g in state_scores[state_name]["passed_singles"][:8]:
                rows.append({
                    "Feature": g["feature"],
                    "Direction": gate_operator_text(g["direction"]),
                    "Threshold": fmt_num(g["threshold"], 3),
                    "Current": fmt_num(g["current"], 3),
                    "Hit Rate": fmt_num(g["hit_rate"] * 100, 1),
                    "Lift": fmt_num(g["lift"], 2),
                })
            if rows:
                st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
            else:
                st.write("No gates currently passing.")


if __name__ == "__main__":
    main()
