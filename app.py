#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Breadth Quant Engine Ultimate v8
Adds oscillator-aware repair logic on top of historical gates/range map.
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
st.set_page_config(page_title="Breadth Quant Engine Ultimate v8", layout="wide", page_icon="📈")

CUSTOM_CSS = """
<style>
:root{--bg:#0b1020;--panel:#111936;--text:#ecf2ff;--muted:#98abd5;--green:#22c55e;--yellow:#f59e0b;--red:#ef4444;--blue:#38bdf8;}
.block-container{padding-top:1rem;padding-bottom:2rem;}
.main-title{padding:1rem 1.2rem;border-radius:18px;background:linear-gradient(135deg, rgba(56,189,248,.18), rgba(167,139,250,.18));border:1px solid rgba(148,163,184,.22);margin-bottom:1rem;}
.soft-card{background:linear-gradient(180deg, rgba(17,25,54,.96), rgba(10,17,38,.98));border:1px solid rgba(148,163,184,.24);border-radius:18px;padding:1rem;box-shadow:0 10px 35px rgba(0,0,0,.22);margin-bottom:1rem;}
.score-title{color:#bcd0ff;font-size:1.02rem;font-weight:800;}
.score-value{font-size:2.7rem;font-weight:950;color:#fff;margin:.35rem 0;}
.pill{display:inline-block;padding:.3rem .6rem;border-radius:999px;font-size:.82rem;font-weight:700;border:1px solid rgba(255,255,255,.12);margin-right:.35rem;}
.pill-green{background:rgba(34,197,94,.16);color:#bbf7d0;}
.pill-yellow{background:rgba(245,158,11,.16);color:#fde68a;}
.pill-red{background:rgba(239,68,68,.16);color:#fecaca;}
.small-muted{color:#93a4cc;font-size:.88rem;}
.setup-line{padding:.45rem .55rem;border-radius:10px;margin:.28rem 0;border:1px solid rgba(255,255,255,.08);}
.setup-pass{background:rgba(34,197,94,.10);}
.setup-near{background:rgba(245,158,11,.10);}
.setup-far{background:rgba(239,68,68,.10);}
.bar{height:8px;border-radius:999px;background:rgba(255,255,255,.08);overflow:hidden;margin-top:.25rem;}
.fill-green{height:100%;background:linear-gradient(90deg,#22c55e,#4ade80);}
.fill-yellow{height:100%;background:linear-gradient(90deg,#f59e0b,#fbbf24);}
.fill-red{height:100%;background:linear-gradient(90deg,#ef4444,#f87171);}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown("""
<div class='main-title'>
  <div style='font-size:1.7rem;font-weight:900;'>📈 Breadth Quant Engine Ultimate v8</div>
  <div class='small-muted'>Historical gates + range map + oscillator-aware repair matrix + practical trade setup parameters.</div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Paths / constants
# -----------------------------
APP_DIR = Path("breadth_quant_store_v8")
APP_DIR.mkdir(exist_ok=True)
HIST_DAILY_PATH = APP_DIR / "daily_history.parquet"
HIST_WEEKLY_PATH = APP_DIR / "weekly_history.parquet"
MODEL_PATH = APP_DIR / "learned_model.json"

KEY_FEATURES = [
    "$BPSPX", "$BPSPX_%B", "$BPNYA", "$OEXA200R", "$SPXA50R", "$NYMO", "$NYSI",
    "$CPCE", "$NYHL", "$NYAD", "$SPXADP", "$TRIN", "$VIX", "RSP:SPY"
]
INVERSE_INDICATORS = {"$TRIN", "$VIX", "$CPCE"}
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
    "_oexa200r": "$OEXA200R", "oexa200r": "$OEXA200R", "_spxa50r": "$SPXA50R", "spxa50r": "$SPXA50R",
    "_nymo": "$NYMO", "nymo": "$NYMO", "_nysi": "$NYSI", "nysi": "$NYSI",
    "_cpce": "$CPCE", "cpce": "$CPCE", "_nyhl": "$NYHL", "nyhl": "$NYHL",
    "_nyad": "$NYAD", "nyad": "$NYAD", "_spxadp": "$SPXADP", "spxadp": "$SPXADP",
    "_trin": "$TRIN", "trin": "$TRIN", "_vix": "$VIX", "vix": "$VIX",
    "hyg_ief": "HYG:IEF", "rsp_spy": "RSP:SPY", "smh_spy": "SMH:SPY",
    "iwm_spy": "IWM:SPY", "xlf_spy": "XLF:SPY", "spxs_svol": "SPXS:SVOL",
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
    return series.ewm(span=span, adjust=False).mean()


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


def macd_hist(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    line = ema(series, fast) - ema(series, slow)
    sig = ema(line, signal)
    return line - sig


def stoch_from_close(close: pd.Series, length: int = 14, smoothk: int = 3) -> pd.Series:
    lo = close.rolling(length).min()
    hi = close.rolling(length).max()
    denom = (hi - lo).replace(0, np.nan)
    k = 100 * (close - lo) / denom
    return k.rolling(smoothk).mean()


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    tp = (high + low + close) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))


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
        rows.append({"date": dt, "open": nums[0], "high": nums[1], "low": nums[2], "close": nums[3], "volume": nums[4]})
    if not rows:
        raise ValueError("No rows parsed from StockCharts CSV")
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def symbol_from_filename(name: str) -> Tuple[str, str]:
    stem = Path(name).stem.strip().lower()
    timeframe = "weekly" if stem.endswith("w") or stem.endswith("_w") else "daily"
    stem = stem.replace("w", "").replace("_w", "").strip("_")
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
    weekly_df = pd.concat(weekly, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True) if weekly else pd.DataFrame(columns=daily_df.columns)
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
                gates.append({"feature": feat, "direction": direction, "threshold": thr, "support": support, "hit_rate": hit, "base_rate": base, "lift": hit/base if base > 0 else np.nan, "score": score})
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
            combos.append({"gates": [g1, g2], "support": support, "hit_rate": hit, "base_rate": base, "lift": hit/base if base > 0 else np.nan, "score": score})
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


def band_distance_score(x: float, q25: float, med: float, q75: float) -> float:
    if pd.isna(x) or pd.isna(q25) or pd.isna(med) or pd.isna(q75):
        return np.nan
    iqr = max(abs(q75 - q25), 1e-6)
    if q25 <= x <= q75:
        d = abs(x - med) / iqr
        return max(0.72, 1.0 - 0.28 * d)
    d = min(abs(x - med) / iqr, 3.0)
    return max(0.0, 0.72 - 0.24 * (d - 1.0))


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
        rows.append({
            "cluster": int(cl),
            "samples": int((labels == cl).sum()),
            "bounce_rate": float(base.loc[sub_idx, "bounce_success"].mean()),
            "repair_rate": float(base.loc[sub_idx, "repair_success"].mean()),
            "regime_rate": float(base.loc[sub_idx, "regime_success"].mean()),
            "fall_rate": float(base.loc[sub_idx, "fall_success"].mean()),
            **{f"{f}_median": float(base.loc[sub_idx, f].median()) for f in features if f in base.columns}
        })
    stats_df = pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)
    names = assign_cluster_names(stats_df)
    stats_df["cluster_name"] = stats_df["cluster"].map(names)
    art = ClusterArtifacts(scaler_mean=scaler.mean_.tolist(), scaler_scale=scaler.scale_.tolist(), features=features, centroids=km.cluster_centers_.tolist(), cluster_names={str(k): v for k, v in names.items()}, silhouette_score=float(sil))
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
        close_row = piv_close.loc[latest_date].copy()
        for sym, val in snap_map.items():
            if sym in close_row.index:
                close_row[sym] = val
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
        rows.append({
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
        })
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
        rows.append({
            "Feature": feat, "Current": cur,
            "Bounce Range": f"{fmt_num(b25,3)} – {fmt_num(b75,3)}", "Bounce Center": bc,
            "Repair Range": f"{fmt_num(r25,3)} – {fmt_num(r75,3)}", "Repair Center": rc,
            "Regime Range": f"{fmt_num(g25,3)} – {fmt_num(g75,3)}", "Regime Center": gc,
            "State Ladder": state,
        })
    return pd.DataFrame(rows)


def nearest_confirmation_from_ranges(snapshot: Dict[str, float], bands: Dict[str, Any], bullish: bool = True) -> List[dict]:
    setups = []
    target_state = "repair" if bullish else "fall"
    target_bands = bands.get(target_state, {}) if target_state != "fall" else {}
    if bullish:
        for feat in ["$BPSPX_%B", "$SPXA50R", "$BPSPX", "$BPNYA", "$NYMO", "$CPCE", "$TRIN", "RSP:SPY"]:
            cur = safe_float(snapshot.get(feat, np.nan))
            if pd.isna(cur):
                continue
            if feat in bands.get("repair", {}):
                q25 = bands["repair"][feat]["q25"]
                q75 = bands["repair"][feat]["q75"]
                if feat in INVERSE_INDICATORS:
                    target = q75 if cur > q75 else q25
                    op = "≤" if cur > q75 else "inside"
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
                q75 = bands["bounce"][feat]["q75"]
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
    long_flag = (((bounce["pass_frac"] >= 0.55 and bounce["prob"] >= max(0.40, bounce["base_rate"] + 0.08)) or (repair["pass_frac"] >= 0.50 and repair["prob"] >= max(0.30, repair["base_rate"] + 0.06))) and recovery_score >= 45 and repairing_count >= 3 and canary["label"] != "Risk-Off") or ((regime["pass_frac"] >= 0.50 and regime["prob"] >= max(0.30, regime["base_rate"] + 0.05) and canary["label"] != "Risk-Off"))
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


@st.cache_data(show_spinner=False)
def build_model_from_history_bytes(file_bytes: bytes) -> Dict[str, Any]:
    daily, weekly = parse_stockcharts_zip(file_bytes)
    return build_model_from_history(daily, weekly)


def load_model() -> Optional[Dict[str, Any]]:
    return load_json(MODEL_PATH, None)


def render_score_card(title: str, value: float, pct: Optional[float] = None):
    st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='score-title'>{title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='score-value'>{int(round(value)) if pd.notna(value) else 'n/a'}</div>", unsafe_allow_html=True)
    if pct is not None and pd.notna(pct):
        st.markdown(f"<span class='pill pill-red'>{pct:.1f}%</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_signal_box(signal: str, text: str):
    klass = "pill-green" if signal == "LONG" else "pill-red" if signal == "SHORT" else "pill-yellow"
    st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='score-title'>Daily Verdict</div><div class='score-value'>{signal}</div><span class='pill {klass}'>{text}</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def main():
    with st.sidebar:
        st.header("⚙️ Configuration")
        hist_upload = st.file_uploader("Historical ZIP", type=["zip"])
        snap_upload = st.file_uploader("Daily Snapshot CSV", type=["csv"])
        force_rebuild = st.toggle("Force rebuild model", value=False)
        run_backtest = st.button("Run historical backtest")
        reset = st.button("Reset Model")
        if reset:
            for p in [HIST_DAILY_PATH, HIST_WEEKLY_PATH, MODEL_PATH]:
                if p.exists():
                    p.unlink()
            st.success("Model reset")
            st.rerun()

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
    band_df, band_totals = score_bands(snapshot, model.get("bands", {}))
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

    # simple recovery score from oscillator matrix
    recovery_score = float(np.clip(osc_df["Repair Score"].mean() / 4 * 100, 0, 100)) if not osc_df.empty else np.nan

    cluster_info = model.get("clusters")
    cluster_name, cluster_conf = None, None
    if cluster_info:
        cl_id, cluster_name, cluster_conf = predict_cluster(snapshot, ClusterArtifacts(
            scaler_mean=cluster_info["mean"], scaler_scale=cluster_info["scale"], features=cluster_info["features"],
            centroids=cluster_info["centroids"], cluster_names=cluster_info["names"], silhouette_score=cluster_info.get("silhouette", 0.0)
        ))

    signal = classify_signal(state_scores, canary, recovery_score, cluster_name, osc_df)
    long_setups = nearest_confirmation_from_ranges(snapshot, model.get("bands", {}), bullish=True)
    short_setups = nearest_confirmation_from_ranges(snapshot, model.get("bands", {}), bullish=False)

    c1, c2, c3, c4 = st.columns(4)
    with c1: render_score_card("Bounce", 100 * state_scores["bounce"]["prob"] if pd.notna(state_scores["bounce"]["prob"]) else 0, 100 * state_scores["bounce"]["prob"] if pd.notna(state_scores["bounce"]["prob"]) else np.nan)
    with c2: render_score_card("Repair", 100 * state_scores["repair"]["prob"] if pd.notna(state_scores["repair"]["prob"]) else 0, 100 * state_scores["repair"]["prob"] if pd.notna(state_scores["repair"]["prob"]) else np.nan)
    with c3: render_score_card("Regime", 100 * state_scores["regime"]["prob"] if pd.notna(state_scores["regime"]["prob"]) else 0, 100 * state_scores["regime"]["prob"] if pd.notna(state_scores["regime"]["prob"]) else np.nan)
    with c4: render_score_card("Fall", 100 * state_scores["fall"]["prob"] if pd.notna(state_scores["fall"]["prob"]) else 0, 100 * state_scores["fall"]["prob"] if pd.notna(state_scores["fall"]["prob"]) else np.nan)

    render_signal_box(signal["signal"], " | ".join(signal["reasons"][:4]))

    tab1, tab2, tab3, tab4 = st.tabs(["Decision Dashboard", "Range Map / State Ladder", "Backtest vs Buy & Hold", "Diagnostics"])
    with tab1:
        st.markdown("<div class='soft-card'><div class='score-title'>Intraday / Repair Context</div>", unsafe_allow_html=True)
        proxy = proxy_nymo(snapshot, prev_snapshot)
        st.write(f"NYMO Proxy: {fmt_num(proxy['value'])} | Delta: {fmt_num(proxy['delta'])} | State: {proxy['state']}")
        st.write(f"Recovery Score: {fmt_num(recovery_score,1)} | Canary: {canary['label']} | Cluster: {cluster_name or 'n/a'}")
        st.markdown("</div>", unsafe_allow_html=True)

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
        if run_backtest:
            st.info("Backtest placeholder in v8. Main focus here is upgraded gate/range/oscillator logic.")
        else:
            st.write("Click 'Run historical backtest' in the sidebar to run backtest logic.")

    with tab4:
        st.markdown("<div class='soft-card'><div class='score-title'>Band Alignment</div>", unsafe_allow_html=True)
        st.dataframe(band_df, width='stretch', hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div class='soft-card'><div class='score-title'>Current Snapshot</div>", unsafe_allow_html=True)
        snap_show = pd.DataFrame({"Feature": sorted([k for k in snapshot.keys() if k in KEY_FEATURES or any(k.startswith(f) for f in OSC_FEATURES)]), "Value": [snapshot[k] for k in sorted([k for k in snapshot.keys() if k in KEY_FEATURES or any(k.startswith(f) for f in OSC_FEATURES)])]})
        st.dataframe(snap_show, width='stretch', hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
