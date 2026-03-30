#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import math
import zipfile
from dataclasses import dataclass, asdict
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# =============================================================================
# App config
# =============================================================================
st.set_page_config(
    page_title="Breadth Quant Engine Ultimate v2",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root{
  --bg:#0b1020;--panel:#111936;--panel2:#162246;--text:#ecf2ff;--muted:#98abd5;
  --green:#22c55e;--yellow:#f59e0b;--red:#ef4444;--blue:#38bdf8;--purple:#a78bfa;
}
.block-container{padding-top:1rem;padding-bottom:2rem;}
.main-title{padding:1rem 1.2rem;border-radius:18px;background:linear-gradient(135deg, rgba(56,189,248,.18), rgba(167,139,250,.18));border:1px solid rgba(148,163,184,.22);margin-bottom:1rem;}
.soft-card{background:linear-gradient(180deg, rgba(17,25,54,.96), rgba(10,17,38,.98));border:1px solid rgba(148,163,184,.24);border-radius:18px;padding:1rem;box-shadow:0 10px 35px rgba(0,0,0,.22);}
.score-title{color:#bcd0ff;font-size:1.0rem;font-weight:800;}
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
<h1 style="margin:0;font-size:1.8rem;">📈 Breadth Quant Engine Ultimate v2</h1>
<p style="margin:0.5rem 0 0 0;color:#98abd5;">Hard historical gates + improvement patterns + canary + cluster + on-demand backtest</p>
</div>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# Storage
# =============================================================================
APP_DIR = Path("breadth_quant_store_ultimate_v2")
APP_DIR.mkdir(exist_ok=True)
DAILY_PATH = APP_DIR / "daily_history.parquet"
WEEKLY_PATH = APP_DIR / "weekly_history.parquet"
MODEL_PATH = APP_DIR / "model.json"
SNAPSHOT_DIR = APP_DIR / "snapshots"
SNAPSHOT_DIR.mkdir(exist_ok=True)

# =============================================================================
# Constants
# =============================================================================
EASTERN = "America/New_York"
TREND_WINDOWS = [1, 2, 3, 5, 10]
KEY_FEATURES = [
    "$BPSPX", "$BPSPX_%B", "$BPNYA", "$OEXA200R", "$SPXA50R", "$NYMO", "$NYSI",
    "$CPCE", "$NYHL", "$NYAD", "$SPXADP", "$TRIN", "$VIX", "RSP:SPY"
]
INVERSE_INDICATORS = {"$TRIN", "$VIX", "$CPCE", "VXX"}
CANARY_WEIGHTS = {
    "SPXS:SVOL": 0.24,
    "HYG:IEF": 0.20,
    "SMH:SPY": 0.18,
    "XLF:SPY": 0.12,
    "RSP:SPY": 0.12,
    "IWM:SPY": 0.10,
    "$VIX": 0.04,
}
OUTCOME_DEFS = {
    "bounce": {"horizon": 10, "ret": 0.03, "dd": -0.03, "type": "max"},
    "repair": {"horizon": 20, "ret": 0.04, "dd": -0.05, "type": "end"},
    "regime": {"horizon": 60, "ret": 0.08, "dd": -0.08, "type": "end"},
    "fall": {"horizon": 10, "ret": -0.03, "dd": 0.03, "type": "min_end"},
}

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


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, default=str))


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def score_color(score: float, max_score: float = 100.0) -> str:
    frac = 0.0 if max_score == 0 else max(0.0, min(1.0, score / max_score))
    if frac >= 0.7:
        return "pill-green"
    if frac >= 0.4:
        return "pill-yellow"
    return "pill-red"


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
        return "Midday"
    if t < dt_time(16, 0):
        return "Late Day"
    if t < dt_time(18, 0):
        return "Post Close"
    return "Official EOD"

# =============================================================================
# Indicators
# =============================================================================
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


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


def cci(close: pd.Series, window: int = 20) -> pd.Series:
    sma = close.rolling(window).mean()
    mad = (close - sma).abs().rolling(window).mean()
    return (close - sma) / (0.015 * mad.replace(0, np.nan))


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
        g["rsi14"] = rsi(close, 14)
        g["pct_b20"] = percent_b(close, 20, 2.0)
        g["macd_hist"] = macd_hist(close)
        g["cci20"] = cci(close, 20)
        g["stoch"] = stoch_from_close(close, 14, 3)
        for w in TREND_WINDOWS:
            g[f"d{w}"] = close.diff(w)
            g[f"roc{w}"] = 100 * (close / close.shift(w) - 1)
            g[f"rsi_d{w}"] = g["rsi14"].diff(w)
            g[f"cci_d{w}"] = g["cci20"].diff(w)
            g[f"pb_d{w}"] = g["pct_b20"].diff(w)
        out.append(g)
    return pd.concat(out, ignore_index=True)

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
        rows.append({"date": dt, "open": nums[0], "high": nums[1], "low": nums[2], "close": nums[3], "volume": nums[4]})
    if not rows:
        raise ValueError("No rows parsed from StockCharts CSV")
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def symbol_from_filename(name: str) -> Tuple[str, str]:
    stem = Path(name).stem.strip().lower()
    timeframe = "weekly" if stem.endswith("w") or stem.endswith("_w") else "daily"
    stem = stem.replace("_w", "").rstrip("w").strip("_")
    mapping = {
        "rsp": "RSP", "ursp": "URSP", "spy": "SPY", "vxx": "VXX",
        "bpspx": "$BPSPX", "bpnya": "$BPNYA", "oexa200r": "$OEXA200R", "spxa50r": "$SPXA50R",
        "nymo": "$NYMO", "nysi": "$NYSI", "cpce": "$CPCE", "nyhl": "$NYHL", "nyad": "$NYAD",
        "spxadp": "$SPXADP", "trin": "$TRIN", "vix": "$VIX",
        "hyg_ief": "HYG:IEF", "rsp_spy": "RSP:SPY", "smh_spy": "SMH:SPY", "iwm_spy": "IWM:SPY",
        "xlf_spy": "XLF:SPY", "spxs_svol": "SPXS:SVOL",
    }
    return mapping.get(stem, stem.upper()), timeframe


def parse_stockcharts_zip(file_bytes: bytes) -> Tuple[pd.DataFrame, pd.DataFrame]:
    daily, weekly = [], []
    with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
        for name in zf.namelist():
            if name.endswith("/") or not name.lower().endswith(".csv"):
                continue
            try:
                df = parse_stockcharts_csv(zf.read(name))
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
    cols = {c.lower().strip(): c for c in df.columns}
    if "symbol" not in cols:
        raise ValueError("Snapshot needs a Symbol column")
    symbol_col = cols["symbol"]
    close_col = cols.get("close") or cols.get("last") or cols.get("price") or cols.get("daily close")
    if close_col is None:
        raise ValueError("Snapshot needs a Close/Price column")
    out = pd.DataFrame({
        "Symbol": df[symbol_col].astype(str).str.strip(),
        "Close": pd.to_numeric(df[close_col], errors="coerce"),
    })
    pct_candidates = [c for c in df.columns if "pct" in c.lower() or "%" in c.lower()]
    out["PctChange"] = pd.to_numeric(df[pct_candidates[0]], errors="coerce") if pct_candidates else np.nan
    return out.dropna(subset=["Symbol"])

# =============================================================================
# Outcomes + model learning
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
    end_ret = rets[:, -1].copy()

    valid_any = np.isfinite(rets).any(axis=1)
    max_gain = np.full(n, np.nan)
    max_dd = np.full(n, np.nan)
    if valid_any.any():
        max_gain[valid_any] = np.nanmax(rets[valid_any], axis=1)
        max_dd[valid_any] = np.nanmin(rets[valid_any], axis=1)

    valid = n - horizon
    end_ret[valid:] = np.nan
    max_gain[valid:] = np.nan
    max_dd[valid:] = np.nan
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


def direction_hints(feature: str, state: str) -> List[str]:
    if state == "fall":
        return ["gte", "lte"] if feature in INVERSE_INDICATORS else ["lte", "gte"]
    if feature in INVERSE_INDICATORS:
        return ["lte", "gte"] if state in {"repair", "regime"} else ["gte", "lte"]
    return ["gte", "lte"] if state in {"repair", "regime"} else ["lte", "gte"]


def learn_single_gates(df: pd.DataFrame, features: List[str], label: str, min_support: int = 40) -> List[dict]:
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
                gates.append({
                    "feature": feat,
                    "direction": direction,
                    "threshold": thr,
                    "support": support,
                    "hit_rate": hit,
                    "base_rate": base,
                    "lift": hit / base if base > 0 else np.nan,
                    "score": score,
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
                "gates": [g1, g2],
                "support": support,
                "hit_rate": hit,
                "base_rate": base,
                "lift": hit / base if base > 0 else np.nan,
                "score": score,
            })
    combos = sorted(combos, key=lambda x: (x["score"], x["lift"], x["support"]), reverse=True)
    return combos[:6]


def summarize_bands(df: pd.DataFrame, label: str, features: List[str]) -> Dict[str, Dict[str, float]]:
    hit = df[df[f"{label}_success"] == 1.0]
    res = {}
    for feat in features:
        if feat not in hit.columns:
            continue
        s = pd.to_numeric(hit[feat], errors="coerce").dropna()
        if len(s) < 10:
            continue
        res[feat] = {
            "median": float(s.median()),
            "q25": float(s.quantile(0.25)),
            "q75": float(s.quantile(0.75)),
            "count": int(len(s)),
        }
    return res

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
        bpspx_pb = row.get("$BPSPX_%B_median", np.nan)
        if fall >= max(repair, bounce, regime) and fall > 0.30:
            names[idx] = "Deterioration cluster"
        elif regime >= max(repair, bounce, fall) and regime > 0.30:
            names[idx] = "Durable regime"
        elif repair >= max(regime, bounce, fall) and repair > 0.22:
            names[idx] = "Repair cluster"
        elif bounce >= max(regime, repair, fall) and bounce > 0.35:
            names[idx] = "Bounce cluster"
        elif pd.notna(bpspx_pb) and bpspx_pb < 0.15:
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
            "bounce_rate": float(outcomes_df.loc[sub.index, "bounce_success"].mean()),
            "repair_rate": float(outcomes_df.loc[sub.index, "repair_success"].mean()),
            "regime_rate": float(outcomes_df.loc[sub.index, "regime_success"].mean()),
            "fall_rate": float(outcomes_df.loc[sub.index, "fall_success"].mean()),
            **{f"{feat}_median": float(sub[feat].median()) for feat in features},
        })
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
    confidence = 1.0 / (1.0 + float(dists[cl]))
    return cl, artifacts.cluster_names.get(str(cl), f"Cluster {cl}"), confidence


def build_canary_from_history(daily_feat: pd.DataFrame) -> pd.DataFrame:
    piv = daily_feat.pivot(index="date", columns="symbol", values="close")
    score_map = {}
    for sym in CANARY_WEIGHTS:
        if sym not in piv.columns:
            continue
        s = piv[sym].dropna()
        if len(s) < 220:
            continue
        mh = macd_hist(s)
        tsi = s.diff().ewm(span=20).mean() / s.diff().abs().ewm(span=20).mean() * 100
        stoch = stoch_from_close(s, 14, 3)
        cci100 = ((s - s.rolling(100).mean()) / (0.015 * (s - s.rolling(100).mean()).abs().rolling(100).mean().replace(0, np.nan)))
        df = pd.concat([mh.rename("macdh"), tsi.rename("tsi"), stoch.rename("stoch"), cci100.rename("cci")], axis=1).dropna()
        if df.empty:
            continue
        bull = (df["macdh"] > 0) & (df["tsi"] > 0) & (df["stoch"] > 50) & (df["cci"] > 0)
        bear = (df["macdh"] < 0) & (df["tsi"] < 0) & (df["stoch"] < 50) & (df["cci"] < 0)
        sc = pd.Series(0.0, index=df.index)
        sc[bull] = 1.0
        sc[bear] = -1.0
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
    out = pd.DataFrame({"date": df.index, "canary_comp": comp.values, "canary_conf": conf.values})
    return out.reset_index(drop=True)


def compute_recovery_momentum(piv: pd.DataFrame, latest_date) -> Dict[str, Any]:
    features = ["$NYSI", "$NYMO", "$BPSPX", "$SPXA50R", "$NYHL", "$TRIN", "$CPCE", "$VIX", "RSP:SPY"]
    comps = {}
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
        x_mean, y_mean = np.mean(x), np.mean(y)
        denom = np.sum((x - x_mean) ** 2)
        slope = 0.0 if denom == 0 else np.sum((x - x_mean) * (y - y_mean)) / denom
        scale = max(np.nanmean(np.abs(y)), 1e-6)
        comps[feat] = slope / scale
    for feat in INVERSE_INDICATORS:
        if feat in comps and pd.notna(comps[feat]):
            comps[feat] = -comps[feat]
    vals = [np.tanh(v * 3.0) for v in comps.values() if pd.notna(v)]
    score = np.nan if not vals else float(np.mean(vals)) * 50 + 50
    score = np.nan if pd.isna(score) else max(0.0, min(100.0, score))
    return {"recovery_score": score, "components": comps}


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


def gate_pass(cur: float, gate: dict) -> bool:
    if pd.isna(cur):
        return False
    return cur >= gate["threshold"] if gate["direction"] == "gte" else cur <= gate["threshold"]


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
        "passed_combos": [x for x in combos if x["passed"]],
        "all_singles": singles,
    }


def build_hold_setup(state_scores: Dict[str, Any], snapshot: Dict[str, float]) -> Dict[str, Any]:
    long_triggers, short_triggers = [], []
    # nearest unpassed gates
    for label in ["bounce", "repair", "regime"]:
        failed = [x for x in state_scores[label]["all_singles"] if not x["passed"] and pd.notna(x["current"])]
        if failed:
            failed = sorted(failed, key=lambda x: abs(x["current"] - x["threshold"]))
            for g in failed[:2]:
                op = "≥" if g["direction"] == "gte" else "≤"
                long_triggers.append(f"{g['feature']} {op} {fmt_num(g['threshold'],3)} (now {fmt_num(g['current'],3)})")
    failed_fall = [x for x in state_scores["fall"]["all_singles"] if not x["passed"] and pd.notna(x["current"])]
    if failed_fall:
        failed_fall = sorted(failed_fall, key=lambda x: abs(x["current"] - x["threshold"]))
        for g in failed_fall[:3]:
            op = "≥" if g["direction"] == "gte" else "≤"
            short_triggers.append(f"{g['feature']} {op} {fmt_num(g['threshold'],3)} (now {fmt_num(g['current'],3)})")
    return {
        "long_setup": list(dict.fromkeys(long_triggers))[:5],
        "short_setup": list(dict.fromkeys(short_triggers))[:5],
    }


def classify_signal(state_scores: Dict[str, Any], band_totals: Dict[str, float], canary: Dict[str, Any], cluster_name: Optional[str], recovery_score: float) -> Dict[str, Any]:
    bounce, repair, regime, fall = state_scores["bounce"], state_scores["repair"], state_scores["regime"], state_scores["fall"]
    signal = "HOLD"
    reasons = []
    long_flag = (
        (((bounce["pass_frac"] >= 0.55 and bounce["prob"] >= max(0.40, bounce["base_rate"] + 0.08)) or
          (repair["pass_frac"] >= 0.50 and repair["prob"] >= max(0.30, repair["base_rate"] + 0.06))) and recovery_score >= 45 and canary["label"] != "Risk-Off")
        or (regime["pass_frac"] >= 0.50 and regime["prob"] >= max(0.30, regime["base_rate"] + 0.05) and canary["label"] != "Risk-Off")
    )
    short_flag = (fall["pass_frac"] >= 0.50 and fall["prob"] >= max(0.25, fall["base_rate"] + 0.05) and canary["label"] != "Risk-On" and recovery_score < 40)
    if long_flag:
        signal = "LONG"
    elif short_flag:
        signal = "SHORT"
    else:
        signal = "HOLD"
    if signal == "LONG":
        reasons.append(f"Bounce/Repair probability favorable ({max(bounce['prob'], repair['prob']):.0%})")
        reasons.append(f"Recovery momentum: {recovery_score:.0f}")
        reasons.append(f"Canary: {canary['label']}")
    elif signal == "SHORT":
        reasons.append(f"Fall probability dominant ({fall['prob']:.0%})")
        reasons.append(f"Recovery momentum weak: {recovery_score:.0f}")
        reasons.append(f"Canary: {canary['label']}")
    else:
        reasons.append("Mixed signals; no strong edge")
    if cluster_name:
        reasons.append(f"Cluster: {cluster_name}")
    hold_setup = build_hold_setup(state_scores, {}) if signal == "HOLD" else {"long_setup": [], "short_setup": []}
    return {
        "signal": signal,
        "reasons": reasons,
        "bounce_prob": bounce["prob"],
        "repair_prob": repair["prob"],
        "regime_prob": regime["prob"],
        "fall_prob": fall["prob"],
        "hold_setup": hold_setup,
    }


def calculate_risk_metrics(equity_curve: pd.Series, rf: float = 0.02) -> Dict[str, float]:
    returns = equity_curve.pct_change().dropna()
    if len(returns) < 20:
        return {"sharpe": np.nan, "sortino": np.nan, "calmar": np.nan, "max_dd": np.nan, "total_return": np.nan, "cagr": np.nan}
    excess = returns - rf / 252
    sharpe = (excess.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    downside = returns[returns < 0]
    sortino = (excess.mean() / downside.std()) * np.sqrt(252) if len(downside) > 0 and downside.std() > 0 else 0
    peak = equity_curve.cummax()
    dd = equity_curve / peak - 1
    max_dd = dd.min()
    calmar = (returns.mean() * 252) / abs(max_dd) if max_dd != 0 and not pd.isna(max_dd) else 0
    years = len(equity_curve) / 252
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1 if years > 0 else 0
    return {"sharpe": sharpe, "sortino": sortino, "calmar": calmar, "max_dd": max_dd, "total_return": equity_curve.iloc[-1] / equity_curve.iloc[0] - 1, "cagr": cagr}


def run_backtest(score_df: pd.DataFrame, fast: int = 5, slow: int = 13, deadband: float = 0.0, use_canary: bool = True, canary_thr: float = 0.05, conf_thr: float = 55.0, switch_cost_bps: float = 5.0) -> pd.DataFrame:
    bt = score_df.copy().sort_values("date").reset_index(drop=True)
    if "master_score" not in bt.columns or "rsp_close" not in bt.columns:
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
    bt["spy_ret"] = bt["spy_close"].pct_change().fillna(0) if "spy_close" in bt.columns else 0.0
    bt["turnover"] = bt["signal"].diff().abs().fillna(0)
    cost = switch_cost_bps / 10000.0
    bt["strategy_ret"] = bt["signal"] * bt["ret"] - bt["turnover"] * cost
    bt["equity_strategy"] = (1 + bt["strategy_ret"]).cumprod()
    bt["equity_rsp_buyhold"] = (1 + bt["ret"]).cumprod()
    if "spy_close" in bt.columns:
        bt["equity_spy_buyhold"] = (1 + bt["spy_ret"]).cumprod()
    return bt

@st.cache_data(show_spinner=False)
def build_model_cached(file_bytes: bytes) -> Dict[str, Any]:
    daily, weekly = parse_stockcharts_zip(file_bytes)
    return build_model_from_history(daily, weekly)


def build_model_from_history(daily: pd.DataFrame, weekly: pd.DataFrame) -> Dict[str, Any]:
    daily_feat = add_indicator_features(daily)
    weekly_feat = add_indicator_features(weekly) if not weekly.empty else weekly.copy()
    piv = daily_feat.pivot(index="date", columns="symbol", values="close")
    if "RSP" not in piv.columns:
        raise ValueError("RSP daily history is required")
    outcomes = build_outcomes(piv["RSP"].dropna())
    base = piv.join(outcomes, how="inner").dropna()

    # Add improvement features from key breadth series if present
    for sym in ["$BPSPX", "$NYMO", "$NYSI", "$SPXA50R", "$NYHL", "$TRIN", "$CPCE", "$VIX"]:
        if sym in piv.columns:
            s = piv[sym]
            for w in TREND_WINDOWS:
                base[f"{sym}_d{w}"] = s.diff(w)
                base[f"{sym}_roc{w}"] = 100 * (s / s.shift(w) - 1)
    # Ratios if available
    if "RSP:SPY" in piv.columns:
        s = piv["RSP:SPY"]
        for w in TREND_WINDOWS:
            base[f"RSP:SPY_d{w}"] = s.diff(w)
    features = [c for c in base.columns if c not in ["RSP"] + [f"{x}_success" for x in OUTCOME_DEFS.keys()]]
    features = [c for c in features if base[c].notna().sum() >= 80]

    learned = {"states": {}, "bands": {}, "meta": {"rows": int(len(base))}}
    for state in ["bounce", "repair", "regime", "fall"]:
        singles = learn_single_gates(base, features, state)
        combos = learn_combo_gates(base, state, singles)
        learned["states"][state] = {"singles": singles, "combos": combos, "base_rate": float(base[f"{state}_success"].mean())}
        if state != "fall":
            learned["bands"][state] = summarize_bands(base, state, KEY_FEATURES)

    cluster_base = base.dropna(subset=[k for k in KEY_FEATURES if k in base.columns]).copy()
    cluster_features = [k for k in KEY_FEATURES if k in cluster_base.columns]
    cluster_stats, cluster_artifacts = build_clusters(cluster_base, cluster_features, n_clusters=6)
    canary_hist = build_canary_from_history(daily_feat)
    learned["clusters"] = {
        "features": cluster_artifacts.features,
        "mean": cluster_artifacts.scaler_mean,
        "scale": cluster_artifacts.scaler_scale,
        "centroids": cluster_artifacts.centroids,
        "names": cluster_artifacts.cluster_names,
        "silhouette": cluster_artifacts.silhouette_score,
        "stats": cluster_stats.to_dict(orient="records"),
    }
    learned["canary_hist"] = canary_hist.to_dict(orient="records") if not canary_hist.empty else []
    save_json(MODEL_PATH, learned)
    daily_feat.to_parquet(DAILY_PATH, index=False)
    if not weekly_feat.empty:
        weekly_feat.to_parquet(WEEKLY_PATH, index=False)
    return learned


def load_model() -> Optional[Dict[str, Any]]:
    if not MODEL_PATH.exists() or not DAILY_PATH.exists():
        return None
    return load_json(MODEL_PATH, None)


def build_snapshot_from_history(daily_feat: pd.DataFrame) -> Tuple[pd.Timestamp, Dict[str, float], Dict[str, float], pd.DataFrame]:
    piv = daily_feat.pivot(index="date", columns="symbol", values="close")
    latest_date = pd.to_datetime(daily_feat["date"]).max()
    prior_date = pd.to_datetime(daily_feat[daily_feat["date"] < latest_date]["date"]).max()
    snapshot, prior_snapshot = {}, {}
    for feat in KEY_FEATURES:
        sym = feat
        if feat.endswith("_%B"):
            base_sym = feat.replace("_%B", "")
            sym_rows = daily_feat[daily_feat["symbol"] == base_sym].sort_values("date")
            if not sym_rows.empty:
                series = sym_rows.set_index("date")["pct_b20"] if "pct_b20" in sym_rows.columns else pd.Series(dtype=float)
                snapshot[feat] = safe_float(series.get(latest_date, np.nan))
                prior_snapshot[feat] = safe_float(series.get(prior_date, np.nan))
        else:
            if sym in piv.columns:
                snapshot[feat] = safe_float(piv.loc[latest_date, sym])
                prior_snapshot[feat] = safe_float(piv.loc[prior_date, sym]) if pd.notna(prior_date) else np.nan
            else:
                snapshot[feat] = np.nan
                prior_snapshot[feat] = np.nan
    return latest_date, snapshot, prior_snapshot, piv


def apply_snapshot_override(daily_feat: pd.DataFrame, snapshot_df: pd.DataFrame) -> Tuple[pd.Timestamp, Dict[str, float], Dict[str, float], pd.DataFrame]:
    daily_feat = daily_feat.copy()
    snap_map = {str(r["Symbol"]).strip(): safe_float(r["Close"]) for _, r in snapshot_df.iterrows() if pd.notna(safe_float(r["Close"]))}
    for sym, new_close in snap_map.items():
        mask = daily_feat["symbol"] == sym
        if mask.any():
            idx = daily_feat.loc[mask, "date"].idxmax()
            daily_feat.at[idx, "close"] = new_close
            if "high" in daily_feat.columns:
                daily_feat.at[idx, "high"] = max(safe_float(daily_feat.at[idx, "high"]), new_close)
            if "low" in daily_feat.columns:
                daily_feat.at[idx, "low"] = min(safe_float(daily_feat.at[idx, "low"]), new_close)
    daily_feat = add_indicator_features(daily_feat[[c for c in daily_feat.columns if c in {"date","open","high","low","close","volume","symbol"}]])
    return build_snapshot_from_history(daily_feat)


def render_score_card(title: str, value: float) -> None:
    pill = score_color(0 if pd.isna(value) else value)
    st.markdown(
        f"<div class='soft-card'><div class='score-title'>{title}</div><div class='score-value'>{fmt_num(value,0)}</div><span class='pill {pill}'>{title}</span></div>",
        unsafe_allow_html=True,
    )


def render_signal_box(signal: str, subtitle: str) -> None:
    cls = "signal-hold"
    if signal == "LONG":
        cls = "signal-long"
    elif signal == "SHORT":
        cls = "signal-short"
    st.markdown(f"<div class='{cls}'><div class='score-title'>Daily Verdict</div><div class='score-value'>{signal}</div><div>{subtitle}</div></div>", unsafe_allow_html=True)


def main() -> None:
    with st.sidebar:
        st.header("⚙️ Configuration")
        phase = detect_session_phase()
        st.write(f"Session: {phase}")
        use_proxy = st.toggle("Use proxy NYMO intraday", value=(phase != "Official EOD"))
        force_rebuild = st.toggle("Force rebuild model", value=False)
        st.subheader("🛡️ Canary")
        use_canary = st.toggle("Enable canary", value=True)
        canary_thr = st.slider("Canary threshold", -0.20, 0.30, 0.05, 0.01)
        conf_thr = st.slider("Confidence threshold", 0, 100, 55)
        st.subheader("📊 Backtest")
        fast_ema = st.slider("Fast EMA", 3, 15, 5)
        slow_ema = st.slider("Slow EMA", 8, 34, 13)
        deadband = st.slider("Deadband", 0.0, 0.10, 0.0, 0.005)
        switch_cost = st.slider("Switch cost (bps)", 0.0, 25.0, 5.0, 0.5)
        hist_upload = st.file_uploader("Historical ZIP", type=["zip"])
        snap_upload = st.file_uploader("Daily Snapshot CSV", type=["csv"])
        if st.button("Reset Model", width="stretch"):
            for p in [DAILY_PATH, WEEKLY_PATH, MODEL_PATH]:
                if p.exists():
                    p.unlink()
            st.success("Model reset")
            st.rerun()

    model = None
    if hist_upload is not None and (force_rebuild or not MODEL_PATH.exists()):
        with st.spinner("Building historical model..."):
            model = build_model_cached(hist_upload.read())
        st.sidebar.success("Model built")
    if model is None:
        model = load_model()
    if model is None:
        st.info("Upload your historical ZIP to build the model.")
        return

    daily_feat = pd.read_parquet(DAILY_PATH)
    if snap_upload is not None:
        snap_df = parse_snapshot_csv(snap_upload.read())
        latest_date, snapshot, prior_snapshot, piv = apply_snapshot_override(daily_feat, snap_df)
    else:
        latest_date, snapshot, prior_snapshot, piv = build_snapshot_from_history(daily_feat)

    nymo_eff = proxy_nymo(snapshot, prior_snapshot) if use_proxy else {"value": safe_float(snapshot.get("$NYMO", np.nan)), "delta": safe_float(snapshot.get("$NYMO", np.nan)) - safe_float(prior_snapshot.get("$NYMO", np.nan)), "state": "Official"}

    state_scores = {state: evaluate_state(snapshot, model["states"][state]) for state in ["bounce", "repair", "regime", "fall"]}
    band_df, band_totals = score_bands(snapshot, model.get("bands", {}))
    recovery = compute_recovery_momentum(piv, latest_date)
    recovery_score = recovery["recovery_score"]

    canary_hist = pd.DataFrame(model.get("canary_hist", []))
    if not canary_hist.empty:
        canary_hist["date"] = pd.to_datetime(canary_hist["date"])
        row = canary_hist[canary_hist["date"] <= latest_date].tail(1)
        canary_comp = float(row["canary_comp"].iloc[0]) if not row.empty else 0.0
        canary_conf = float(row["canary_conf"].iloc[0]) if not row.empty else 0.0
    else:
        canary_comp, canary_conf = 0.0, 0.0
    canary_label = "Risk-On" if canary_comp > canary_thr and canary_conf >= conf_thr else "Risk-Off" if canary_comp < -abs(canary_thr) else "Neutral"
    canary = {"label": canary_label, "comp": canary_comp, "conf": canary_conf}

    cluster_name = None
    cluster_conf = None
    if model.get("clusters"):
        art = ClusterArtifacts(
            scaler_mean=model["clusters"]["mean"],
            scaler_scale=model["clusters"]["scale"],
            features=model["clusters"]["features"],
            centroids=model["clusters"]["centroids"],
            cluster_names=model["clusters"]["names"],
            silhouette_score=model["clusters"].get("silhouette", 0.0),
        )
        _, cluster_name, cluster_conf = predict_cluster(snapshot, art)

    signal = classify_signal(state_scores, band_totals, canary, cluster_name, 0 if pd.isna(recovery_score) else recovery_score)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_score_card("Bounce", (state_scores["bounce"]["prob"] or 0) * 100 if pd.notna(state_scores["bounce"]["prob"]) else 0)
    with c2:
        render_score_card("Repair", (state_scores["repair"]["prob"] or 0) * 100 if pd.notna(state_scores["repair"]["prob"]) else 0)
    with c3:
        render_score_card("Regime", (state_scores["regime"]["prob"] or 0) * 100 if pd.notna(state_scores["regime"]["prob"]) else 0)
    with c4:
        render_score_card("Fall", (state_scores["fall"]["prob"] or 0) * 100 if pd.notna(state_scores["fall"]["prob"]) else 0)

    subtitle = f"Canary: {canary['label']} | Recovery: {fmt_num(recovery_score,0)} | Cluster: {cluster_name or 'n/a'}"
    render_signal_box(signal["signal"], subtitle)

    tab1, tab2, tab3 = st.tabs(["Decision Dashboard", "Backtest vs Buy & Hold", "Diagnostics"])

    with tab1:
        a, b = st.columns([1.2, 1])
        with a:
            st.markdown("### Why")
            for r in signal["reasons"]:
                st.markdown(f"- {r}")
            st.markdown("### Intraday Proxy Context")
            st.markdown(f"- NYMO Proxy: {fmt_num(nymo_eff['value'],2)}")
            st.markdown(f"- Delta: {fmt_num(nymo_eff['delta'],2)}")
            st.markdown(f"- State: {nymo_eff['state']}")
            st.markdown(f"- Recovery Score: {fmt_num(recovery_score,1)}")
            st.markdown(f"- Cluster Confidence: {fmt_num(cluster_conf,2)}")
        with b:
            st.markdown("### Hold Trade Setup Parameters")
            if signal["signal"] == "HOLD":
                st.markdown("**Go LONG if these start to trigger:**")
                if signal["hold_setup"]["long_setup"]:
                    for item in signal["hold_setup"]["long_setup"]:
                        st.markdown(f"- {item}")
                else:
                    st.markdown("- No clean nearby long trigger found")
                st.markdown("**Go SHORT if these start to trigger:**")
                if signal["hold_setup"]["short_setup"]:
                    for item in signal["hold_setup"]["short_setup"]:
                        st.markdown(f"- {item}")
                else:
                    st.markdown("- No clean nearby short trigger found")
            else:
                st.markdown("- Not in HOLD mode")
            st.markdown("### Current Snapshot")
            snap_show = pd.DataFrame({"Feature": list(snapshot.keys()), "Value": list(snapshot.values())})
            st.dataframe(snap_show, width="stretch", hide_index=True)

    with tab2:
        st.markdown("### Historical Strategy vs Buy & Hold")
        run_bt = st.button("Run historical backtest", width="stretch")
        if run_bt:
            with st.spinner("Running backtest..."):
                piv_hist = daily_feat.pivot(index="date", columns="symbol", values="close")
                dates = sorted(piv_hist.index.unique())
                rows = []
                cluster_art = None
                if model.get("clusters"):
                    cluster_art = ClusterArtifacts(
                        scaler_mean=model["clusters"]["mean"], scaler_scale=model["clusters"]["scale"],
                        features=model["clusters"]["features"], centroids=model["clusters"]["centroids"],
                        cluster_names=model["clusters"]["names"], silhouette_score=model["clusters"].get("silhouette", 0.0),
                    )
                for i, dt in enumerate(dates):
                    hist_sub = daily_feat[daily_feat["date"] <= dt]
                    if hist_sub.empty:
                        continue
                    ld, snap, prev_snap, piv_sub = build_snapshot_from_history(hist_sub)
                    rec = compute_recovery_momentum(piv_sub, ld)
                    can_label = "Neutral"
                    if not canary_hist.empty:
                        row = canary_hist[canary_hist["date"] <= dt].tail(1)
                        if not row.empty:
                            cc = float(row["canary_comp"].iloc[0])
                            cf = float(row["canary_conf"].iloc[0])
                            can_label = "Risk-On" if cc > canary_thr and cf >= conf_thr else "Risk-Off" if cc < -abs(canary_thr) else "Neutral"
                            can = {"label": can_label, "comp": cc, "conf": cf}
                        else:
                            can = {"label": "Neutral", "comp": 0.0, "conf": 0.0}
                    else:
                        can = {"label": "Neutral", "comp": 0.0, "conf": 0.0}
                    cl_name = None
                    if cluster_art is not None:
                        _, cl_name, _ = predict_cluster(snap, cluster_art)
                    ss = {state: evaluate_state(snap, model["states"][state]) for state in ["bounce","repair","regime","fall"]}
                    _, btot = score_bands(snap, model.get("bands", {}))
                    sig = classify_signal(ss, btot, can, cl_name, 0 if pd.isna(rec["recovery_score"]) else rec["recovery_score"])
                    bounce_prob = ss["bounce"]["prob"] if pd.notna(ss["bounce"]["prob"]) else 0
                    repair_prob = ss["repair"]["prob"] if pd.notna(ss["repair"]["prob"]) else 0
                    regime_prob = ss["regime"]["prob"] if pd.notna(ss["regime"]["prob"]) else 0
                    fall_prob = ss["fall"]["prob"] if pd.notna(ss["fall"]["prob"]) else 0
                    master = 0.35*bounce_prob + 0.30*repair_prob + 0.25*regime_prob - 0.20*fall_prob
                    rows.append({
                        "date": dt,
                        "master_score": master,
                        "rsp_close": piv_hist.loc[dt, "RSP"] if "RSP" in piv_hist.columns else np.nan,
                        "spy_close": piv_hist.loc[dt, "SPY"] if "SPY" in piv_hist.columns else np.nan,
                        "canary_comp": can.get("comp",0.0),
                        "canary_conf": can.get("conf",0.0),
                        "signal_text": sig["signal"],
                    })
                bt_input = pd.DataFrame(rows).dropna(subset=["rsp_close"])
                bt = run_backtest(bt_input, fast_ema, slow_ema, deadband, use_canary, canary_thr, conf_thr, switch_cost)
                if bt.empty:
                    st.warning("Backtest could not run.")
                else:
                    strat_metrics = calculate_risk_metrics(bt["equity_strategy"])
                    rsp_metrics = calculate_risk_metrics(bt["equity_rsp_buyhold"])
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Strategy Return", fmt_num(strat_metrics["total_return"]*100,1)+"%")
                        st.metric("Strategy Sharpe", fmt_num(strat_metrics["sharpe"],2))
                    with c2:
                        st.metric("RSP Buy&Hold", fmt_num(rsp_metrics["total_return"]*100,1)+"%")
                        st.metric("RSP Sharpe", fmt_num(rsp_metrics["sharpe"],2))
                    with c3:
                        if "equity_spy_buyhold" in bt.columns:
                            spy_metrics = calculate_risk_metrics(bt["equity_spy_buyhold"])
                            st.metric("SPY Buy&Hold", fmt_num(spy_metrics["total_return"]*100,1)+"%")
                            st.metric("SPY Sharpe", fmt_num(spy_metrics["sharpe"],2))
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=bt["date"], y=bt["equity_strategy"], name="Strategy"))
                    fig.add_trace(go.Scatter(x=bt["date"], y=bt["equity_rsp_buyhold"], name="RSP Buy&Hold"))
                    if "equity_spy_buyhold" in bt.columns:
                        fig.add_trace(go.Scatter(x=bt["date"], y=bt["equity_spy_buyhold"], name="SPY Buy&Hold"))
                    fig.update_layout(title="Equity Curves", height=500)
                    st.plotly_chart(fig, width="stretch")
                    st.dataframe(bt.tail(20), width="stretch")

    with tab3:
        st.markdown("### Gate Diagnostics")
        for state in ["bounce", "repair", "regime", "fall"]:
            st.markdown(f"**{state.title()}** — Prob {fmt_num((state_scores[state]['prob'] or 0)*100,1)}% | Pass frac {fmt_num(state_scores[state]['pass_frac']*100,1)}%")
            passed = state_scores[state]["passed_singles"][:6]
            if passed:
                for g in passed:
                    op = "≥" if g["direction"] == "gte" else "≤"
                    st.markdown(f"- {g['feature']} {op} {fmt_num(g['threshold'],3)} | now {fmt_num(g['current'],3)} | hit {fmt_num(g['hit_rate']*100,1)}%")
        st.markdown("### Band Alignment")
        st.dataframe(band_df, width="stretch")
        st.markdown("### Recovery Components")
        rec_df = pd.DataFrame({"Feature": list(recovery["components"].keys()), "Slope": list(recovery["components"].values())})
        st.dataframe(rec_df, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
