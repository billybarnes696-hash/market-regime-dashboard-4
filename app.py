
#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import logging
import math
import re
import traceback
import zipfile
from dataclasses import dataclass, asdict
from datetime import datetime, time as dt_time
from enum import Enum
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

# =============================
# App / logging
# =============================
APP_NAME = "Breadth Historical Gate Engine PRO+"
APP_DIR = Path("breadth_quant_store_v2")
APP_DIR.mkdir(exist_ok=True)
SNAPSHOT_DIR = APP_DIR / "snapshots"
SNAPSHOT_DIR.mkdir(exist_ok=True)
HIST_DAILY_PATH = APP_DIR / "daily_history.parquet"
HIST_WEEKLY_PATH = APP_DIR / "weekly_history.parquet"
MODEL_PATH = APP_DIR / "learned_model.json"
UPLOAD_HISTORY_PATH = APP_DIR / "upload_history.csv"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title=APP_NAME, layout="wide", page_icon="📈")

st.markdown(
    """
    <style>
    :root{
      --bg:#0b1020;--panel:#111936;--panel2:#162246;--text:#ecf2ff;--muted:#98abd5;
      --green:#22c55e;--yellow:#f59e0b;--red:#ef4444;--blue:#38bdf8;--purple:#a78bfa;
    }
    .block-container{padding-top:1rem;padding-bottom:2rem;}
    .main-title{
      padding:1rem 1.2rem;border-radius:18px;
      background:linear-gradient(135deg, rgba(56,189,248,.18), rgba(167,139,250,.18));
      border:1px solid rgba(148,163,184,.22);margin-bottom:1rem;
    }
    .soft-card{
      background:linear-gradient(180deg, rgba(17,25,54,.96), rgba(10,17,38,.98));
      border:1px solid rgba(148,163,184,.24);border-radius:18px;padding:1rem;
      box-shadow:0 10px 35px rgba(0,0,0,.22);
    }
    .score-title{color:#bcd0ff;font-size:1.02rem;font-weight:800;letter-spacing:.02em;}
    .score-value{font-size:3rem;line-height:1.0;font-weight:950;color:#fff;margin:.35rem 0;}
    .pill{display:inline-block;padding:.28rem .6rem;border-radius:999px;font-size:.82rem;font-weight:700;border:1px solid rgba(255,255,255,.12);margin-right:.35rem;}
    .pill-green{background:rgba(34,197,94,.16);color:#bbf7d0;}
    .pill-yellow{background:rgba(245,158,11,.16);color:#fde68a;}
    .pill-red{background:rgba(239,68,68,.16);color:#fecaca;}
    .pill-blue{background:rgba(56,189,248,.16);color:#bae6fd;}
    .small-muted{color:#93a4cc;font-size:.88rem;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <div class="main-title">
      <div style="font-size:1.65rem;font-weight:900;">📈 {APP_NAME}</div>
      <div class="small-muted">Hard historical gates + repair-pattern momentum + clustering + canary + timestamp-aware intraday proxy governance.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================
# Constants
# =============================
EASTERN = "America/New_York"
INVERSE_INDICATORS = ["$TRIN", "$VIX", "$CPCE", "VXX"]
KEY_LEVEL_FEATURES = [
    "$BPSPX", "$BPSPX_%B", "$BPNYA", "$OEXA200R", "$SPXA50R", "$NYMO", "$NYSI",
    "$CPCE", "$NYHL", "$NYAD", "$SPXADP", "$TRIN", "$VIX", "RSP:SPY"
]
RECOVERY_FEATURES = ["$NYSI", "$NYMO", "$BPSPX_%B", "$SPXA50R", "$NYHL", "$TRIN", "$CPCE", "$VIX", "RSP:SPY"]
WEEKLY_FEATURES = ["$BPSPX", "$SPXA50R", "$NYSI", "$OEXA200R", "RSP"]
CANARY_WEIGHTS = {
    "SPXS:SVOL": 0.20, "HYG:IEF": 0.18, "SMH:SPY": 0.16, "XLF:SPY": 0.12,
    "RSP:SPY": 0.12, "IWM:SPY": 0.10, "SPY:VXX": 0.07, "$VIX": 0.05,
}
OUTCOME_DEFS = {
    "bounce": {"horizon": 10, "ret": 0.03, "dd": -0.03, "type": "max"},
    "repair": {"horizon": 20, "ret": 0.04, "dd": -0.05, "type": "end"},
    "regime": {"horizon": 60, "ret": 0.08, "dd": -0.08, "type": "end"},
    "fall": {"horizon": 10, "ret": -0.03, "dd": 0.03, "type": "min_end"},
}
RECOVERY_WINDOWS = {1:1, 2:2, 3:3, 5:5, 10:10}

class SessionPhase(str, Enum):
    PRE = "Pre-Market"
    OPEN = "Opening Window"
    MID = "Midday Window"
    LATE = "Late-Day Window"
    POST = "Post-Close"
    OFFICIAL = "Official EOD"
    WEEKEND = "Weekend"

# =============================
# Utilities
# =============================
def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return np.nan

def fmt_num(v: Any, d: int = 2) -> str:
    if pd.isna(v):
        return "n/a"
    return f"{float(v):.{d}f}"

def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default

def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, default=str))

def color_for_signal(signal: str) -> str:
    return {"LONG":"pill-green", "SHORT":"pill-red"}.get(signal, "pill-yellow")

def detect_session_phase(now_ts: Optional[pd.Timestamp] = None) -> SessionPhase:
    ts = now_ts or pd.Timestamp.now(tz=EASTERN)
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

# =============================
# Parsing
# =============================
def parse_stockcharts_csv(content: bytes) -> pd.DataFrame:
    text = content.decode("utf-8", errors="ignore")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    rows = []
    for ln in lines:
        if ln.lower().startswith(("date,", "symbol,", "ticker,")):
            continue
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 5:
            continue
        dt = pd.to_datetime(parts[0], errors="coerce")
        if pd.isna(dt):
            continue
        nums = [safe_float(x) for x in parts[1:6]]
        rows.append({
            "date": dt, "open": nums[0], "high": nums[1], "low": nums[2], "close": nums[3],
            "volume": nums[4] if len(nums) > 4 else np.nan,
        })
    if not rows:
        raise ValueError("No rows parsed from StockCharts CSV.")
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

def symbol_from_filename(name: str) -> Tuple[str, str]:
    stem = Path(name).stem.strip().lower()
    tf = "weekly" if stem.endswith(" w") or stem.endswith("_w") or stem.endswith("w") else "daily"
    stem = stem.replace(" w", "").replace("_w", "").strip("_")
    mapping = {
        "rsp": "RSP", "ursp": "URSP", "spy": "SPY", "vxx": "VXX",
        "vix": "$VIX", "_vix": "$VIX", "trin": "$TRIN", "_trin": "$TRIN",
        "_bpspx": "$BPSPX", "bpspx": "$BPSPX", "_bpnya": "$BPNYA", "bpnya": "$BPNYA",
        "_oexa200r": "$OEXA200R", "oexa200r": "$OEXA200R", "_oexa150r": "$OEXA150R", "oexa150r": "$OEXA150R",
        "_oexa50r": "$OEXA50R", "oexa50r": "$OEXA50R", "_spxa50r": "$SPXA50R", "spxa50r": "$SPXA50R",
        "_nymo": "$NYMO", "nymo": "$NYMO", "_nysi": "$NYSI", "nysi": "$NYSI",
        "_cpce": "$CPCE", "cpce": "$CPCE", "_nyhl": "$NYHL", "nyhl": "$NYHL",
        "_nyad": "$NYAD", "nyad": "$NYAD", "_spxadp": "$SPXADP", "spxadp": "$SPXADP",
        "rsp_spy": "RSP:SPY", "smh_spy": "SMH:SPY", "iwm_spy": "IWM:SPY",
        "xlf_spy": "XLF:SPY", "spxs_svol": "SPXS:SVOL", "hyg_ief": "HYG:IEF", "spy_vxx": "SPY:VXX",
    }
    return mapping.get(stem, stem.upper()), tf

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
            except Exception as e:
                logger.warning("skip %s: %s", name, e)
    if not daily:
        raise ValueError("No daily CSV files parsed from ZIP.")
    daily_df = pd.concat(daily, ignore_index=True).sort_values(["symbol","date"]).reset_index(drop=True)
    weekly_df = pd.concat(weekly, ignore_index=True).sort_values(["symbol","date"]).reset_index(drop=True) if weekly else pd.DataFrame(columns=daily_df.columns)
    return daily_df, weekly_df

def parse_snapshot_csv(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    cols = {c.lower(): c for c in df.columns}
    if "symbol" not in cols:
        raise ValueError("Snapshot needs a Symbol column.")
    symbol_col = cols["symbol"]
    close_col = next((cols[c] for c in ["close", "last", "price", "value"] if c in cols), None)
    if close_col is None:
        raise ValueError("Snapshot needs a Close/Last/Price column.")
    pct_col = next((cols[c] for c in ["daily pctchange", "daily pctchange(1,daily close)", "% change", "pctchange"] if c in cols), None)
    out = pd.DataFrame({
        "Symbol": df[symbol_col].astype(str).str.strip(),
        "Close": pd.to_numeric(df[close_col], errors="coerce"),
    })
    if pct_col is not None:
        out["PctChange"] = pd.to_numeric(df[pct_col], errors="coerce")
    return out.dropna(subset=["Close"])

# =============================
# Indicators
# =============================
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def rsi(s: pd.Series, period: int = 14) -> pd.Series:
    delta = s.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = up.ewm(alpha=1/period, adjust=False).mean() / down.ewm(alpha=1/period, adjust=False).mean().replace(0, np.nan)
    return (100 - 100/(1 + rs)).fillna(50)

def percent_b(s: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    ma = s.rolling(window).mean()
    std = s.rolling(window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return (s - lower) / (upper - lower).replace(0, np.nan)

def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    tp = (high + low + close) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))

def roc(s: pd.Series, period: int = 3) -> pd.Series:
    return 100 * (s / s.shift(period) - 1)

def add_indicator_features(hist: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for sym, g in hist.groupby("symbol", sort=False):
        g = g.sort_values("date").copy()
        h, l, c = g["high"], g["low"], g["close"]
        g["rsi14"] = rsi(c, 14)
        g["pct_b20"] = percent_b(c, 20, 2.0)
        g["cci20"] = cci(h, l, c, 20)
        g["roc3"] = roc(c, 3)
        g["slope3"] = c - c.shift(3)
        g["ma20"] = c.rolling(20).mean()
        g["ma50"] = c.rolling(50).mean()
        frames.append(g)
    return pd.concat(frames, ignore_index=True)

def features_wide(hist_feat: pd.DataFrame, feats: List[str]) -> pd.DataFrame:
    parts = []
    for feat in feats:
        p = hist_feat.pivot(index="date", columns="symbol", values=feat)
        p.columns = [f"{col}__{feat}" for col in p.columns]
        parts.append(p)
    return pd.concat(parts, axis=1).sort_index()

def build_snapshot_from_history(daily_feat: pd.DataFrame, snapshot_df: Optional[pd.DataFrame] = None) -> Tuple[Dict[str, float], Dict[str, float], pd.DataFrame]:
    hist = daily_feat.copy()
    prior_date = hist["date"].max()
    if snapshot_df is not None and not snapshot_df.empty:
        rt_map = dict(zip(snapshot_df["Symbol"], snapshot_df["Close"]))
        for sym, new_close in rt_map.items():
            mask = hist["symbol"] == sym
            if mask.any():
                idx = hist.loc[mask].sort_values("date").index[-1]
                hist.at[idx, "close"] = new_close
                hist.at[idx, "high"] = max(hist.at[idx, "high"], new_close)
                hist.at[idx, "low"] = min(hist.at[idx, "low"], new_close)
    hist_feat = add_indicator_features(hist)
    latest_date = hist_feat["date"].max()
    snapshot, prev_snapshot = {}, {}
    for sym, g in hist_feat.groupby("symbol", sort=False):
        g = g.sort_values("date")
        cur = g.iloc[-1]
        prev = g.iloc[-2] if len(g) > 1 else cur
        snapshot[sym] = safe_float(cur["close"])
        snapshot[f"{sym}_%B"] = safe_float(cur.get("pct_b20"))
        snapshot[f"{sym}_RSI14"] = safe_float(cur.get("rsi14"))
        snapshot[f"{sym}_CCI20"] = safe_float(cur.get("cci20"))
        prev_snapshot[sym] = safe_float(prev["close"])
        prev_snapshot[f"{sym}_%B"] = safe_float(prev.get("pct_b20"))
        prev_snapshot[f"{sym}_RSI14"] = safe_float(prev.get("rsi14"))
        prev_snapshot[f"{sym}_CCI20"] = safe_float(prev.get("cci20"))
    return snapshot, prev_snapshot, hist_feat

# =============================
# Forward outcomes / learning
# =============================
def compute_future_metrics(price: pd.Series, horizon: int):
    vals = price.to_numpy(dtype=float)
    n = len(vals)
    if n < horizon + 1:
        idx = price.index
        return pd.Series(np.nan, index=idx), pd.Series(np.nan, index=idx), pd.Series(np.nan, index=idx)

    idx = np.arange(n)
    window_idx = idx[:, None] + np.arange(1, horizon + 1)
    mask = window_idx >= n
    # Clip before indexing so NumPy never sees out-of-bounds indices.
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
        elif cfg["type"] == "end":
            success = (end_ret >= cfg["ret"]) & (max_dd >= cfg["dd"])
        else:
            success = (end_ret <= cfg["ret"])
        out[f"{name}_success"] = success.astype(float)
    return out

def base_state_table(daily_feat: pd.DataFrame) -> pd.DataFrame:
    piv = daily_feat.pivot(index="date", columns="symbol", values="close")
    req = ["RSP"]
    if not all(r in piv.columns for r in req):
        raise ValueError("RSP history is required.")
    wide = features_wide(daily_feat, ["close", "pct_b20", "rsi14", "cci20", "roc3", "slope3"])
    out = pd.DataFrame(index=wide.index)
    def map_feature(sym, raw):
        col = f"{sym}__{raw}"
        return wide[col] if col in wide.columns else pd.Series(np.nan, index=wide.index)
    out["$BPSPX"] = map_feature("$BPSPX", "close")
    out["$BPSPX_%B"] = map_feature("$BPSPX", "pct_b20")
    out["$BPNYA"] = map_feature("$BPNYA", "close")
    out["$OEXA200R"] = map_feature("$OEXA200R", "close")
    out["$SPXA50R"] = map_feature("$SPXA50R", "close")
    out["$NYMO"] = map_feature("$NYMO", "close")
    out["$NYSI"] = map_feature("$NYSI", "close")
    out["$CPCE"] = map_feature("$CPCE", "close")
    out["$NYHL"] = map_feature("$NYHL", "close")
    out["$NYAD"] = map_feature("$NYAD", "close")
    out["$SPXADP"] = map_feature("$SPXADP", "close")
    out["$TRIN"] = map_feature("$TRIN", "close")
    out["$VIX"] = map_feature("$VIX", "close")
    out["RSP:SPY"] = map_feature("RSP:SPY", "close")
    # derived improvement features
    for feat in ["$NYSI", "$NYMO", "$BPSPX_%B", "$SPXA50R", "$NYHL", "$TRIN", "$CPCE", "$VIX", "RSP:SPY"]:
        for w in [1,2,3,5,10]:
            out[f"{feat}_d{w}"] = out[feat] - out[feat].shift(w)
    for feat in ["$NYSI", "$NYMO", "$BPSPX", "$SPXA50R"]:
        out[f"{feat}_RSI14"] = map_feature(feat.replace("_%B",""), "rsi14")
        out[f"{feat}_CCI20"] = map_feature(feat.replace("_%B",""), "cci20")
    outcomes = build_outcomes(piv["RSP"])
    out = out.join(outcomes, how="left")
    out["rsp_close"] = piv["RSP"]
    out.index.name = "date"
    return out.reset_index()

def direction_hints(feature: str, state: str) -> List[str]:
    if state == "fall":
        if any(k in feature for k in ["$TRIN", "$VIX", "$CPCE"]):
            return ["gte", "lte"]
        return ["lte", "gte"]
    if any(k in feature for k in ["_d", "CCI20", "RSI14"]):
        # improvement features: higher generally better except inverse indicators
        if any(inv in feature for inv in ["$TRIN", "$VIX", "$CPCE"]):
            return ["lte", "gte"]
        return ["gte", "lte"]
    if feature in ["$TRIN", "$VIX", "$CPCE", "VXX"]:
        return ["lte", "gte"] if state in {"repair","regime"} else ["gte","lte"]
    return ["gte","lte"] if state in {"repair","regime"} else ["lte","gte"]

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

def summarize_bands(df: pd.DataFrame, label: str) -> Dict[str, Dict[str, float]]:
    hit = df[df[f"{label}_success"] == 1.0]
    res = {}
    for feat in KEY_LEVEL_FEATURES:
        if feat not in hit.columns:
            continue
        s = pd.to_numeric(hit[feat], errors="coerce").dropna()
        if len(s) < 10:
            continue
        res[feat] = {
            "median": float(s.median()),
            "q25": float(s.quantile(0.25)),
            "q75": float(s.quantile(0.75)),
            "count": int(len(s))
        }
    return res

def build_model_from_history(daily: pd.DataFrame, weekly: pd.DataFrame) -> Dict[str, Any]:
    daily_feat = add_indicator_features(daily)
    weekly_feat = add_indicator_features(weekly) if not weekly.empty else weekly.copy()
    base = base_state_table(daily_feat)
    features = [c for c in base.columns if c not in ["date", "rsp_close", *[f"{x}_success" for x in OUTCOME_DEFS.keys()]]]
    learned = {"states": {}, "bands": {}, "meta": {"rows": int(len(base))}}
    for state in ["bounce", "repair", "regime", "fall"]:
        singles = learn_single_gates(base, features, state)
        combos = learn_combo_gates(base, state, singles)
        learned["states"][state] = {"singles": singles, "combos": combos, "base_rate": float(base[f"{state}_success"].mean())}
        if state != "fall":
            learned["bands"][state] = summarize_bands(base, state)

    # clustering on key features
    cl_df = base.dropna(subset=KEY_LEVEL_FEATURES).copy()
    X_raw = cl_df[KEY_LEVEL_FEATURES].astype(float)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    k = min(6, max(3, len(cl_df)//60))
    km = KMeans(n_clusters=k, random_state=42, n_init=20).fit(X)
    sil = float(silhouette_score(X, km.labels_)) if len(np.unique(km.labels_)) > 1 else 0.0
    cluster_stats = []
    for cl in sorted(np.unique(km.labels_)):
        mask = km.labels_ == cl
        sub = cl_df.loc[mask]
        row = {"cluster": int(cl), "samples": int(mask.sum())}
        for state in ["bounce", "repair", "regime", "fall"]:
            row[f"{state}_rate"] = float(sub[f"{state}_success"].mean())
        row["$BPSPX_%B_median"] = float(sub["$BPSPX_%B"].median())
        cluster_stats.append(row)
    cluster_names = {}
    for row in cluster_stats:
        if row["regime_rate"] > max(row["bounce_rate"], row["repair_rate"], row["fall_rate"]):
            name = "Durable regime"
        elif row["repair_rate"] > max(row["bounce_rate"], row["regime_rate"], row["fall_rate"]):
            name = "Repair cluster"
        elif row["bounce_rate"] > max(row["repair_rate"], row["regime_rate"], row["fall_rate"]):
            name = "Bounce cluster"
        elif row["fall_rate"] >= max(row["bounce_rate"], row["repair_rate"], row["regime_rate"]):
            name = "Deterioration cluster"
        elif row["$BPSPX_%B_median"] < 0.15:
            name = "Capitulation / washout"
        else:
            name = "Mixed / transitional"
        cluster_names[str(row["cluster"])] = name
    learned["clusters"] = {
        "features": KEY_LEVEL_FEATURES,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "centroids": km.cluster_centers_.tolist(),
        "silhouette": sil,
        "names": cluster_names,
        "stats": cluster_stats,
    }

    # weekly overlay gates
    learned["weekly"] = {"singles": [], "combos": [], "base_rate": np.nan}
    if not weekly_feat.empty:
        weekly_wide = features_wide(weekly_feat, ["close", "pct_b20", "rsi14", "cci20"]).reset_index()
        wk = pd.DataFrame({"date": weekly_wide["date"]})
        for sym in WEEKLY_FEATURES:
            col = f"{sym}__close"
            if col in weekly_wide.columns:
                wk[sym] = weekly_wide[col]
        daily_outcomes = base[["date", "regime_success"]].sort_values("date")
        wk = pd.merge_asof(wk.sort_values("date"), daily_outcomes, on="date", direction="forward").dropna(subset=["regime_success"])
        weekly_features_present = [c for c in wk.columns if c not in ["date", "regime_success"]]
        singles = learn_single_gates(wk.rename(columns={"regime_success":"regime_success"}), weekly_features_present, "regime", min_support=8)
        combos = learn_combo_gates(wk.rename(columns={"regime_success":"regime_success"}), "regime", singles, min_support=6)
        learned["weekly"] = {"singles": singles[:8], "combos": combos[:4], "base_rate": float(wk["regime_success"].mean())}
    # canary from history
    piv = daily_feat.pivot(index="date", columns="symbol", values="close")
    canary_rows = []
    if not piv.empty:
        for date in piv.index:
            row_scores = []
            for ratio, w in CANARY_WEIGHTS.items():
                if ratio in piv.columns and pd.notna(piv.loc[date, ratio]):
                    s = piv[ratio].loc[:date].dropna()
                    if len(s) >= 60:
                        fast = ema(s, 12) - ema(s, 26)
                        sig = ema(fast, 9)
                        sc = 1.0 if fast.iloc[-1] > sig.iloc[-1] else -1.0
                        if ratio in ["$VIX"]:
                            sc = -sc
                        row_scores.append((sc, w))
            if row_scores:
                comp = sum(sc*w for sc,w in row_scores) / sum(w for _,w in row_scores)
                conf = 100 * np.mean([abs(sc) for sc,_ in row_scores])
                canary_rows.append({"date": date, "canary_comp": comp, "canary_conf": conf})
    learned["canary_hist"] = canary_rows
    save_json(MODEL_PATH, learned)
    daily_feat.to_parquet(HIST_DAILY_PATH, index=False)
    if not weekly_feat.empty:
        weekly_feat.to_parquet(HIST_WEEKLY_PATH, index=False)
    return learned

# =============================
# Proxy NYMO / NYSI
# =============================
def proxy_nymo(snapshot: Dict[str, float], prev_snapshot: Dict[str, float]) -> Dict[str, float]:
    nyad = safe_float(snapshot.get("$NYAD", np.nan))
    spxadp = safe_float(snapshot.get("$SPXADP", np.nan))
    prev_nyad = safe_float(prev_snapshot.get("$NYAD", np.nan))
    prev_spxadp = safe_float(prev_snapshot.get("$SPXADP", np.nan))
    cur_raw = 0.6 * (0 if pd.isna(nyad) else nyad) + 0.4 * (0 if pd.isna(spxadp) else spxadp)
    prev_raw = 0.6 * (0 if pd.isna(prev_nyad) else prev_nyad) + 0.4 * (0 if pd.isna(prev_spxadp) else prev_spxadp)
    cur = 100 * np.tanh(cur_raw / 1600.0)
    prev = 100 * np.tanh(prev_raw / 1600.0)
    return {"value": float(cur), "delta": float(cur - prev)}

def proxy_nysi(hist_feat: pd.DataFrame, current_snapshot: Dict[str,float]) -> Dict[str, float]:
    # approximate NYSI proxy as rolling cumulative smoothed proxy_nymo
    piv = hist_feat.pivot(index="date", columns="symbol", values="close")
    if "$NYAD" not in piv.columns or "$SPXADP" not in piv.columns:
        return {"value": np.nan, "delta": np.nan}
    raw = 0.6 * piv["$NYAD"].fillna(0) + 0.4 * piv["$SPXADP"].fillna(0)
    prox_nymo = 100 * np.tanh(raw / 1600.0)
    prox_nysi = ema(prox_nymo, 5).cumsum() / 4.0
    val = float(prox_nysi.iloc[-1]) if len(prox_nysi) else np.nan
    prev = float(prox_nysi.iloc[-2]) if len(prox_nysi) > 1 else np.nan
    return {"value": val, "delta": val - prev if pd.notna(prev) else np.nan}

# =============================
# Scoring / verdict
# =============================
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
        "passed_singles": [x for x in singles if x["passed"]], "failed_singles": [x for x in singles if not x["passed"]],
        "passed_combos": [x for x in combos if x["passed"]]
    }

def score_repair_improvement(snapshot: Dict[str, float], prev_snapshot: Dict[str, float]) -> Dict[str, Any]:
    reasons = []
    total = 0
    hits = 0
    for feat in RECOVERY_FEATURES:
        cur = safe_float(snapshot.get(feat, np.nan))
        prev = safe_float(prev_snapshot.get(feat, np.nan))
        if pd.isna(cur) or pd.isna(prev):
            continue
        total += 1
        delta = cur - prev
        bullish = delta < 0 if feat in ["$TRIN", "$VIX", "$CPCE"] else delta > 0
        if bullish:
            hits += 1
            reasons.append(f"{feat} improving")
    # add oscillator features
    for feat in ["$NYSI", "$NYMO", "$BPSPX", "$SPXA50R"]:
        cci_cur = safe_float(snapshot.get(f"{feat}_CCI20", np.nan))
        cci_prev = safe_float(prev_snapshot.get(f"{feat}_CCI20", np.nan))
        if pd.notna(cci_cur) and pd.notna(cci_prev):
            total += 1
            if cci_cur > cci_prev:
                hits += 1
                reasons.append(f"{feat} CCI improving")
        rsi_cur = safe_float(snapshot.get(f"{feat}_RSI14", np.nan))
        rsi_prev = safe_float(prev_snapshot.get(f"{feat}_RSI14", np.nan))
        if pd.notna(rsi_cur) and pd.notna(rsi_prev):
            total += 1
            if rsi_cur > rsi_prev:
                hits += 1
                reasons.append(f"{feat} RSI improving")
    score = 100 * hits / total if total else np.nan
    return {"score": score, "hits": hits, "total": total, "reasons": reasons}

def band_distance_score(x: float, q25: float, med: float, q75: float) -> float:
    if pd.isna(x) or pd.isna(q25) or pd.isna(med) or pd.isna(q75):
        return np.nan
    iqr = max(abs(q75 - q25), 1e-6)
    if q25 <= x <= q75:
        d = abs(x - med) / iqr
        return max(0.72, 1 - 0.28 * d)
    d = min(abs(x - med) / iqr, 3.0)
    return max(0.0, 0.72 - 0.24 * (d - 1))

def score_bands(snapshot: Dict[str, float], bands: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows = []
    totals = {}
    for label in ["bounce", "repair", "regime"]:
        bmap = bands.get(label, {})
        vals = []
        for feat, meta in bmap.items():
            cur = safe_float(snapshot.get(feat, np.nan))
            if feat in ["$TRIN", "$VIX", "$CPCE"] and pd.notna(cur):
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

def current_canary_label(model: Dict[str, Any], latest_date: pd.Timestamp) -> Dict[str, Any]:
    hist = pd.DataFrame(model.get("canary_hist", []))
    if hist.empty:
        return {"label": "Neutral", "comp": np.nan, "conf": np.nan}
    hist["date"] = pd.to_datetime(hist["date"])
    hist = hist.sort_values("date")
    row = hist[hist["date"] <= latest_date].tail(1)
    if row.empty:
        return {"label": "Neutral", "comp": np.nan, "conf": np.nan}
    comp = float(row["canary_comp"].iloc[0])
    conf = float(row["canary_conf"].iloc[0])
    label = "Risk-On" if comp > 0.05 else "Risk-Off" if comp < -0.05 else "Neutral"
    return {"label": label, "comp": comp, "conf": conf}

def predict_cluster(snapshot: Dict[str, float], model: Dict[str, Any]) -> Tuple[Optional[int], Optional[str], Optional[float]]:
    c = model.get("clusters", {})
    feats = c.get("features", [])
    if not feats:
        return None, None, None
    vals = [safe_float(snapshot.get(f, np.nan)) for f in feats]
    if any(pd.isna(v) for v in vals):
        return None, None, None
    x = np.array(vals, dtype=float)
    mean = np.array(c["mean"]); scale = np.where(np.array(c["scale"]) == 0, 1, np.array(c["scale"]))
    scaled = (x - mean) / scale
    cents = np.array(c["centroids"])
    d = np.sqrt(((cents - scaled)**2).sum(axis=1))
    idx = int(np.argmin(d))
    conf = float(1 / (1 + d[idx]))
    return idx, c["names"].get(str(idx), f"Cluster {idx}"), conf

def classify_signal(state_scores: Dict[str, Any], band_totals: Dict[str,float], canary: Dict[str,Any],
                    cluster_name: Optional[str], cluster_conf: Optional[float], weekly_pass_frac: float,
                    improve: Dict[str,Any]) -> Dict[str, Any]:
    bounce, repair, regime, fall = state_scores["bounce"], state_scores["repair"], state_scores["regime"], state_scores["fall"]
    signal = "HOLD"
    reasons = []
    gate_params = {}

    long_flag = (
        (
            (bounce["pass_frac"] >= 0.55 and bounce["prob"] >= max(0.40, bounce["base_rate"] + 0.08)) or
            (repair["pass_frac"] >= 0.50 and repair["prob"] >= max(0.30, repair["base_rate"] + 0.06))
        )
        and improve["score"] >= 45
        and canary["label"] != "Risk-Off"
    ) or (
        regime["pass_frac"] >= 0.50 and regime["prob"] >= max(0.30, regime["base_rate"] + 0.05)
        and weekly_pass_frac >= 0.35 and canary["label"] != "Risk-Off"
    )

    short_flag = (
        fall["pass_frac"] >= 0.50 and fall["prob"] >= max(0.25, fall["base_rate"] + 0.05)
        and canary["label"] != "Risk-On"
        and improve["score"] < 40
    )

    if long_flag:
        signal = "LONG"
    elif short_flag:
        signal = "SHORT"

    def top_reasons(label, sc):
        for g in sc["passed_singles"][:4]:
            op = "≥" if g["direction"] == "gte" else "≤"
            reasons.append(f"{label}: {g['feature']} {op} {g['threshold']:.3f}")
            gate_params[g["feature"]] = {"current": g["current"], "threshold": g["threshold"], "direction": g["direction"]}

    if signal == "LONG":
        if repair["pass_frac"] >= bounce["pass_frac"]:
            top_reasons("Repair", repair)
        else:
            top_reasons("Bounce", bounce)
        reasons.append(f"Improvement breadth: {int(round(improve['score']))}%")
        reasons.append(f"Canary: {canary['label']}")
    elif signal == "SHORT":
        top_reasons("Fall", fall)
        reasons.append(f"Canary: {canary['label']}")
    else:
        reasons.append("Mixed historical gates; no strong edge.")
        reasons.append(f"Bounce {int(round(100*bounce['pass_frac']))}% / Repair {int(round(100*repair['pass_frac']))}% / Regime {int(round(100*regime['pass_frac']))}% / Fall {int(round(100*fall['pass_frac']))}%")
        reasons.append(f"Canary: {canary['label']}")

    if cluster_name:
        reasons.append(f"Cluster: {cluster_name} ({fmt_num(cluster_conf,2)})")

    return {"signal": signal, "reasons": reasons, "gate_params": gate_params}

# =============================
# Rendering
# =============================
def card(title: str, value: str, subtitle: str = ""):
    st.markdown(
        f'<div class="soft-card"><div class="score-title">{title}</div><div class="score-value">{value}</div><div class="small-muted">{subtitle}</div></div>',
        unsafe_allow_html=True,
    )

def plot_symbol(hist_feat: pd.DataFrame, symbol: str):
    g = hist_feat[hist_feat["symbol"] == symbol].sort_values("date").copy()
    if g.empty:
        return None
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.72, 0.28])
    fig.add_trace(go.Scatter(x=g["date"], y=g["close"], name="Close"), row=1, col=1)
    if "ma20" in g.columns:
        fig.add_trace(go.Scatter(x=g["date"], y=g["ma20"], name="MA20"), row=1, col=1)
    if "pct_b20" in g.columns:
        fig.add_trace(go.Scatter(x=g["date"], y=g["pct_b20"], name="%B"), row=2, col=1)
    fig.update_layout(height=500, margin=dict(l=10, r=10, t=30, b=10), template="plotly_white")
    return fig



# =============================
# Historical backtest vs SPY / RSP buy & hold
# =============================
def _signal_to_exposure(sig: str, mode: str = "Long / Hold / Short") -> int:
    if sig == "LONG":
        return 1
    if sig == "SHORT":
        return -1 if mode == "Long / Hold / Short" else 0
    return 0

def run_historical_backtest(model: Dict[str, Any], daily_feat: pd.DataFrame, weekly_feat: pd.DataFrame) -> pd.DataFrame:
    base = base_state_table(daily_feat).sort_values("date").reset_index(drop=True)
    if base.empty:
        return pd.DataFrame()
    # benchmark series
    piv = daily_feat.pivot(index="date", columns="symbol", values="close").sort_index()
    base["spy_close"] = piv["SPY"].reindex(base["date"]).values if "SPY" in piv.columns else np.nan
    base["rsp_close"] = piv["RSP"].reindex(base["date"]).values if "RSP" in piv.columns else base.get("rsp_close", np.nan)

    # prepare weekly snapshots by date
    weekly_snapshots = {}
    if not weekly_feat.empty:
        wk_feat = add_indicator_features(weekly_feat)
        wk_wide = features_wide(wk_feat, ["close"]).reset_index()
        for _, row in wk_wide.iterrows():
            snap = {}
            for sym in WEEKLY_FEATURES:
                col = f"{sym}__close"
                if col in wk_wide.columns:
                    snap[sym] = safe_float(row.get(col, np.nan))
            weekly_snapshots[pd.to_datetime(row["date"])] = snap

    records = []
    for i in range(1, len(base)):
        cur = base.iloc[i].to_dict()
        prev = base.iloc[i - 1].to_dict()

        # state gates
        state_scores = {state: evaluate_state(cur, model["states"][state]) for state in ["bounce", "repair", "regime", "fall"]}
        _, band_totals = score_bands(cur, model.get("bands", {}))
        improve = score_repair_improvement(cur, prev)
        canary = current_canary_label(model, pd.to_datetime(cur["date"]))
        cl_id, cl_name, cl_conf = predict_cluster(cur, model)

        weekly_pass_frac = 0.0
        if model.get("weekly", {}).get("singles") and weekly_snapshots:
            eligible = [d for d in weekly_snapshots.keys() if d <= pd.to_datetime(cur["date"])]
            if eligible:
                wk_date = max(eligible)
                wk_snapshot = weekly_snapshots[wk_date]
                passes = [gate_pass(safe_float(wk_snapshot.get(g["feature"], np.nan)), g) for g in model["weekly"]["singles"]]
                weekly_pass_frac = float(np.mean(passes)) if passes else 0.0

        sig = classify_signal(state_scores, band_totals, canary, cl_name, cl_conf, weekly_pass_frac, improve)["signal"]
        records.append({
            "date": pd.to_datetime(cur["date"]),
            "signal": sig,
            "rsp_close": safe_float(cur.get("rsp_close", np.nan)),
            "spy_close": safe_float(cur.get("spy_close", np.nan)),
            "bounce_prob": state_scores["bounce"]["prob"],
            "repair_prob": state_scores["repair"]["prob"],
            "regime_prob": state_scores["regime"]["prob"],
            "fall_prob": state_scores["fall"]["prob"],
            "improve_score": improve["score"],
        })
    bt = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    if bt.empty:
        return bt
    bt["rsp_ret"] = bt["rsp_close"].pct_change().fillna(0.0)
    bt["spy_ret"] = bt["spy_close"].pct_change().fillna(0.0) if bt["spy_close"].notna().any() else np.nan
    return bt

def finalize_backtest_equity(bt: pd.DataFrame, mode: str = "Long / Hold / Short", switch_cost_bps: float = 2.0) -> pd.DataFrame:
    out = bt.copy()
    if out.empty:
        return out
    out["exposure"] = out["signal"].map(lambda s: _signal_to_exposure(s, mode)).astype(float)
    out["position"] = out["exposure"].shift(1).fillna(0.0)
    out["switch"] = out["position"].diff().abs().fillna(out["position"].abs())
    cost = switch_cost_bps / 10000.0
    out["strategy_ret"] = out["position"] * out["rsp_ret"] - out["switch"] * cost
    out["equity_strategy"] = (1.0 + out["strategy_ret"]).cumprod()
    out["equity_rsp"] = (1.0 + out["rsp_ret"].fillna(0.0)).cumprod()
    if out["spy_ret"].notna().any():
        out["equity_spy"] = (1.0 + out["spy_ret"].fillna(0.0)).cumprod()
    else:
        out["equity_spy"] = np.nan
    return out

def perf_summary(eq: pd.Series, rets: pd.Series) -> Dict[str, float]:
    eq = pd.to_numeric(eq, errors="coerce").dropna()
    rets = pd.to_numeric(rets, errors="coerce").dropna()
    if eq.empty or rets.empty:
        return {"total_return": np.nan, "cagr": np.nan, "max_dd": np.nan, "sharpe": np.nan}
    total_return = float(eq.iloc[-1] - 1.0)
    years = max(len(rets) / 252.0, 1e-9)
    cagr = float(eq.iloc[-1] ** (1 / years) - 1) if eq.iloc[-1] > 0 else np.nan
    dd = eq / eq.cummax() - 1.0
    max_dd = float(dd.min())
    sharpe = float((rets.mean() / rets.std()) * np.sqrt(252)) if rets.std() and not np.isnan(rets.std()) and rets.std() != 0 else np.nan
    return {"total_return": total_return, "cagr": cagr, "max_dd": max_dd, "sharpe": sharpe}
# =============================
# App
# =============================
def load_model() -> Optional[Dict[str, Any]]:
    if not MODEL_PATH.exists() or not HIST_DAILY_PATH.exists():
        return None
    model = load_json(MODEL_PATH, {})
    return model if model else None

def save_snapshot_history(verdict: str, latest_date: pd.Timestamp, probs: Dict[str, Any], snapshot_path: Path):
    hist = pd.read_csv(UPLOAD_HISTORY_PATH) if UPLOAD_HISTORY_PATH.exists() else pd.DataFrame()
    row = {
        "upload_ts": datetime.now().isoformat(),
        "snapshot_file": str(snapshot_path),
        "verdict": verdict,
        "date": str(latest_date.date()),
        "bounce_prob": probs["bounce"],
        "repair_prob": probs["repair"],
        "regime_prob": probs["regime"],
        "fall_prob": probs["fall"],
    }
    hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    hist.to_csv(UPLOAD_HISTORY_PATH, index=False)

def main():
    st.sidebar.header("Data / Model")
    phase = detect_session_phase()
    st.sidebar.write(f"Session phase: **{phase.value}**")
    use_proxy = st.sidebar.toggle("Use proxy NYMO/NYSI before official evening data", value=(phase != SessionPhase.OFFICIAL))
    force_rebuild = st.sidebar.toggle("Force rebuild historical model", value=False)
    reset_model = st.sidebar.button("Reset saved historical model")

    if reset_model:
        for p in [HIST_DAILY_PATH, HIST_WEEKLY_PATH, MODEL_PATH, UPLOAD_HISTORY_PATH]:
            if p.exists():
                p.unlink()
        st.session_state.pop("backtest_cache", None)
        st.sidebar.success("Saved model reset.")

    model = None if force_rebuild else load_model()

    hist_upload = st.sidebar.file_uploader("One-time historical upload (.zip)", type=["zip"])
    if hist_upload is not None:
        try:
            with st.spinner("Building model from historical zip..."):
                daily, weekly = parse_stockcharts_zip(hist_upload.read())
                model = build_model_from_history(daily, weekly)
            st.session_state.pop("backtest_cache", None)
            st.sidebar.success("Historical model built and saved.")
        except Exception as e:
            st.sidebar.error(f"Build failed: {e}")
            st.code(traceback.format_exc())

    if model is None:
        st.info("Upload your historical StockCharts zip once to build the model.")
        return

    daily_feat = pd.read_parquet(HIST_DAILY_PATH)
    latest_hist_date = pd.to_datetime(daily_feat["date"]).max()
    weekly_feat = pd.read_parquet(HIST_WEEKLY_PATH) if HIST_WEEKLY_PATH.exists() else pd.DataFrame()

    snap_upload = st.sidebar.file_uploader("Daily snapshot upload (.csv)", type=["csv"])
    snap_df = None
    if snap_upload is not None:
        try:
            snap_df = parse_snapshot_csv(snap_upload.read())
        except Exception as e:
            st.sidebar.error(f"Snapshot parse failed: {e}")

    snapshot, prev_snapshot, hist_feat_live = build_snapshot_from_history(daily_feat, snap_df)
    latest_date = pd.to_datetime(hist_feat_live["date"]).max()

    nymo_eff = proxy_nymo(snapshot, prev_snapshot) if use_proxy else {
        "value": safe_float(snapshot.get("$NYMO", np.nan)),
        "delta": safe_float(snapshot.get("$NYMO", np.nan)) - safe_float(prev_snapshot.get("$NYMO", np.nan))
    }
    nysi_eff = proxy_nysi(hist_feat_live, snapshot) if use_proxy else {
        "value": safe_float(snapshot.get("$NYSI", np.nan)),
        "delta": safe_float(snapshot.get("$NYSI", np.nan)) - safe_float(prev_snapshot.get("$NYSI", np.nan))
    }
    # override if proxy mode
    if use_proxy:
        snapshot["$NYMO"] = nymo_eff["value"]
        snapshot["$NYSI"] = nysi_eff["value"]

    state_scores = {}
    for state in ["bounce", "repair", "regime", "fall"]:
        state_scores[state] = evaluate_state(snapshot, model["states"][state])

    band_df, band_totals = score_bands(snapshot, model.get("bands", {}))
    improve = score_repair_improvement(snapshot, prev_snapshot)
    canary = current_canary_label(model, latest_date)
    cl_id, cl_name, cl_conf = predict_cluster(snapshot, model)
    weekly_pass_frac = 0.0
    if model.get("weekly", {}).get("singles"):
        weekly_passes = []
        if not weekly_feat.empty:
            wk_feat = add_indicator_features(weekly_feat)
            wk_piv = wk_feat.pivot(index="date", columns="symbol", values="close")
            latest_wk = pd.to_datetime(wk_feat["date"]).max()
            wk_snapshot = {}
            for sym in WEEKLY_FEATURES:
                if sym in wk_piv.columns:
                    wk_snapshot[sym] = safe_float(wk_piv.loc[latest_wk, sym])
            for g in model["weekly"]["singles"]:
                weekly_passes.append(gate_pass(safe_float(wk_snapshot.get(g["feature"], np.nan)), g))
        weekly_pass_frac = float(np.mean(weekly_passes)) if weekly_passes else 0.0

    signal = classify_signal(state_scores, band_totals, canary, cl_name, cl_conf, weekly_pass_frac, improve)

    # Layout
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("Daily Verdict", signal["signal"], f"{phase.value} | {'Proxy' if use_proxy else 'Official'}")
    with c2:
        card("Bounce / Repair", f"{int(round(state_scores['bounce']['prob']*100)) if pd.notna(state_scores['bounce']['prob']) else 'n/a'} / {int(round(state_scores['repair']['prob']*100)) if pd.notna(state_scores['repair']['prob']) else 'n/a'}", "Historical probabilities")
    with c3:
        card("Regime / Fall", f"{int(round(state_scores['regime']['prob']*100)) if pd.notna(state_scores['regime']['prob']) else 'n/a'} / {int(round(state_scores['fall']['prob']*100)) if pd.notna(state_scores['fall']['prob']) else 'n/a'}", "Historical probabilities")
    with c4:
        card("Improvement Breadth", f"{int(round(improve['score'])) if pd.notna(improve['score']) else 'n/a'}%", f"{improve['hits']}/{improve['total']} improving")

    st.markdown(f'<span class="pill {color_for_signal(signal["signal"])}">Signal: {signal["signal"]}</span>'
                f'<span class="pill pill-blue">Canary: {canary["label"]}</span>'
                f'<span class="pill pill-blue">Weekly pass frac: {fmt_num(weekly_pass_frac,2)}</span>'
                f'<span class="pill pill-blue">Cluster: {cl_name or "n/a"} {fmt_num(cl_conf,2) if cl_conf is not None else ""}</span>', unsafe_allow_html=True)

    tabs = st.tabs(["Decision Dashboard", "Learned Gates", "Backtest vs SPY", "History / Uploads"])

    with tabs[0]:
        left, right = st.columns([1.25, 1])
        with left:
            st.markdown('<div class="soft-card"><div class="score-title">Why</div>', unsafe_allow_html=True)
            for r in signal["reasons"]:
                st.markdown(f"- {r}")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="soft-card"><div class="score-title">Intraday Proxy Context</div>', unsafe_allow_html=True)
            st.markdown(f"NYMO: **{fmt_num(nymo_eff['value'],2)}**")
            st.markdown(f"Δ: **{fmt_num(nymo_eff['delta'],2)}**")
            st.markdown(f"NYSI: **{fmt_num(nysi_eff['value'],2)}**")
            st.markdown(f"Δ: **{fmt_num(nysi_eff['delta'],2)}**")
            st.markdown(f"Weekly regime pass fraction: **{fmt_num(weekly_pass_frac*100,0)}%**")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="soft-card"><div class="score-title">Gate Verdicts</div>', unsafe_allow_html=True)
            for state in ["bounce", "repair", "regime", "fall"]:
                s = state_scores[state]
                st.markdown(f"**{state.title()}**")
                st.progress(min(max(s["pass_frac"], 0.0), 1.0), text=f"{int(round(100*s['prob'])) if pd.notna(s['prob']) else 'n/a'}% | Pass {int(round(100*s['pass_frac']))}% | Base {int(round(100*s['base_rate'])) if pd.notna(s['base_rate']) else 'n/a'}%")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="soft-card"><div class="score-title">Gate Inputs</div>', unsafe_allow_html=True)
            gate_params = signal["gate_params"]
            if gate_params:
                for feat, meta in gate_params.items():
                    op = "≥" if meta["direction"] == "gte" else "≤"
                    st.markdown(f"- {feat} = {fmt_num(meta['current'],3)} (gate {op} {fmt_num(meta['threshold'],3)})")
            else:
                st.markdown("No strong gate alignment today.")
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown('<div class="soft-card"><div class="score-title">Current Snapshot Core Readings</div>', unsafe_allow_html=True)
            core = pd.DataFrame(
                [{"Feature": f, "Current": snapshot.get(f, np.nan), "Prior": prev_snapshot.get(f, np.nan), "Δ": safe_float(snapshot.get(f, np.nan)) - safe_float(prev_snapshot.get(f, np.nan))}
                 for f in ["$BPSPX", "$BPSPX_%B", "$BPNYA", "$OEXA200R", "$SPXA50R", "$NYMO", "$NYSI", "$CPCE", "$NYHL", "$NYAD", "$SPXADP", "$TRIN", "$VIX", "RSP:SPY"]]
            )
            st.dataframe(core, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="soft-card"><div class="score-title">Breadth chart panel</div>', unsafe_allow_html=True)
            chart_symbol = st.selectbox("Chart symbol", options=["$SPXA50R", "$BPSPX", "$NYMO", "$NYSI", "RSP", "SPY"], index=0)
            fig = plot_symbol(hist_feat_live, chart_symbol)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with tabs[1]:
        st.markdown('<div class="soft-card"><div class="score-title">Learned Gates</div>', unsafe_allow_html=True)
        gate_state = st.selectbox("State", ["bounce", "repair", "regime", "fall"])
        singles = model["states"][gate_state]["singles"]
        combos = model["states"][gate_state]["combos"]
        if singles:
            sg = pd.DataFrame([{**g, "gate": f"{g['feature']} {'>=' if g['direction']=='gte' else '<='} {round(g['threshold'],3)}"} for g in singles])
            st.subheader("Single gates")
            st.dataframe(sg[["gate","support","hit_rate","base_rate","lift"]], use_container_width=True, hide_index=True)
        if combos:
            cg = pd.DataFrame([{
                "gate": " AND ".join([f"{g['feature']} {'>=' if g['direction']=='gte' else '<='} {round(g['threshold'],3)}" for g in c["gates"]]),
                "support": c["support"], "hit_rate": c["hit_rate"], "base_rate": c["base_rate"], "lift": c["lift"]
            } for c in combos])
            st.subheader("Combo gates")
            st.dataframe(cg, use_container_width=True, hide_index=True)
        st.subheader("Band alignment")
        st.dataframe(band_df, use_container_width=True, hide_index=True)
        st.subheader("Improvement analysis")
        st.write(f"Improvement score: {fmt_num(improve['score'],1)}%")
        for r in improve["reasons"][:20]:
            st.markdown(f"- {r}")
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[2]:
        st.markdown('<div class="soft-card"><div class="score-title">Historical backtest vs SPY / RSP buy & hold</div>', unsafe_allow_html=True)
        mode = st.radio("Strategy mode", ["Long / Hold / Short", "Long / Hold"], horizontal=True)
        switch_cost_bps = st.slider("Switch cost (bps)", 0.0, 25.0, 2.0, 0.5)
        if "backtest_cache" not in st.session_state:
            with st.spinner("Running historical signal backtest..."):
                st.session_state["backtest_cache"] = run_historical_backtest(model, daily_feat, weekly_feat)
        bt_raw = st.session_state["backtest_cache"]
        if bt_raw.empty:
            st.write("Backtest unavailable.")
        else:
            bt = finalize_backtest_equity(bt_raw, mode=mode, switch_cost_bps=switch_cost_bps)
            ssum = perf_summary(bt["equity_strategy"], bt["strategy_ret"])
            rspsum = perf_summary(bt["equity_rsp"], bt["rsp_ret"])
            spysum = perf_summary(bt["equity_spy"], bt["spy_ret"]) if bt["spy_ret"].notna().any() else {"total_return": np.nan, "cagr": np.nan, "max_dd": np.nan, "sharpe": np.nan}

            m1, m2, m3 = st.columns(3)
            with m1:
                card("Strategy Total Return", f"{fmt_num(100*ssum['total_return'],1)}%", f"CAGR {fmt_num(100*ssum['cagr'],1)}% | MaxDD {fmt_num(100*ssum['max_dd'],1)}%")
            with m2:
                card("RSP Buy & Hold", f"{fmt_num(100*rspsum['total_return'],1)}%", f"CAGR {fmt_num(100*rspsum['cagr'],1)}% | MaxDD {fmt_num(100*rspsum['max_dd'],1)}%")
            with m3:
                bench_label = "SPY Buy & Hold" if bt["spy_ret"].notna().any() else "SPY unavailable"
                bench_sub = f"CAGR {fmt_num(100*spysum['cagr'],1)}% | MaxDD {fmt_num(100*spysum['max_dd'],1)}%" if bt["spy_ret"].notna().any() else ""
                card(bench_label, f"{fmt_num(100*spysum['total_return'],1)}%" if bt["spy_ret"].notna().any() else "n/a", bench_sub)

            beat_spy = bt["equity_strategy"].iloc[-1] > bt["equity_spy"].iloc[-1] if bt["spy_ret"].notna().any() else False
            beat_rsp = bt["equity_strategy"].iloc[-1] > bt["equity_rsp"].iloc[-1]
            verdict = []
            verdict.append("beats RSP buy & hold" if beat_rsp else "does not beat RSP buy & hold")
            if bt["spy_ret"].notna().any():
                verdict.append("beats SPY buy & hold" if beat_spy else "does not beat SPY buy & hold")
            st.markdown(f"**Backtest verdict:** strategy {' and '.join(verdict)} over the available historical sample.")

            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(x=bt["date"], y=bt["equity_strategy"], name="Strategy", line=dict(width=2.5)))
            fig_eq.add_trace(go.Scatter(x=bt["date"], y=bt["equity_rsp"], name="RSP Buy & Hold", line=dict(width=2.0)))
            if bt["spy_ret"].notna().any():
                fig_eq.add_trace(go.Scatter(x=bt["date"], y=bt["equity_spy"], name="SPY Buy & Hold", line=dict(width=2.0)))
            fig_eq.update_layout(height=460, margin=dict(l=10, r=10, t=30, b=10), template="plotly_white")
            st.plotly_chart(fig_eq, use_container_width=True)

            show_cols = ["date", "signal", "position", "bounce_prob", "repair_prob", "regime_prob", "fall_prob", "improve_score", "strategy_ret"]
            st.dataframe(bt[show_cols].tail(200), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[3]:
        hist = pd.read_csv(UPLOAD_HISTORY_PATH) if UPLOAD_HISTORY_PATH.exists() else pd.DataFrame()
        st.markdown('<div class="soft-card"><div class="score-title">History / Uploads</div>', unsafe_allow_html=True)
        st.write(f"Using saved historical gate model. Historical upload is not required again unless you want to refresh the model.")
        if not hist.empty:
            st.dataframe(hist.sort_values("upload_ts", ascending=False), use_container_width=True, hide_index=True)
        else:
            st.write("No saved snapshot history yet.")
        if st.button("Save current snapshot & verdict"):
            snap_payload = {
                "date": str(latest_date.date()),
                "snapshot": snapshot,
                "signal": signal,
                "state_probs": {k: state_scores[k]["prob"] for k in state_scores},
            }
            snap_path = SNAPSHOT_DIR / f"snapshot_{latest_date.date().isoformat()}.json"
            save_json(snap_path, snap_payload)
            save_snapshot_history(signal["signal"], latest_date, {k: state_scores[k]["prob"] for k in state_scores}, snap_path)
            st.success(f"Saved {snap_path.name}")
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
