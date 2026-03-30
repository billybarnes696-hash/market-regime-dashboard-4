
#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import logging
import math
import warnings
import zipfile
from dataclasses import dataclass
from datetime import time as dt_time
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

warnings.filterwarnings("ignore", category=RuntimeWarning)

APP_NAME = "Breadth Quant Engine Final"
APP_DIR = Path("breadth_quant_final_store")
APP_DIR.mkdir(exist_ok=True)
SNAPSHOT_DIR = APP_DIR / "snapshots"
SNAPSHOT_DIR.mkdir(exist_ok=True)
HIST_DAILY_PATH = APP_DIR / "daily_history.parquet"
HIST_WEEKLY_PATH = APP_DIR / "weekly_history.parquet"
MODEL_PATH = APP_DIR / "model.json"
UPLOAD_HISTORY_PATH = APP_DIR / "upload_history.csv"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title=APP_NAME, layout="wide", page_icon="📈")
st.markdown(
    """
    <style>
    :root{--bg:#0b1020;--panel:#111936;--text:#ecf2ff;--muted:#98abd5;--green:#22c55e;--yellow:#f59e0b;--red:#ef4444;--blue:#38bdf8;--purple:#a78bfa;}
    .block-container{padding-top:1rem;padding-bottom:2rem;}
    .main-title{padding:1rem 1.2rem;border-radius:18px;background:linear-gradient(135deg, rgba(56,189,248,.18), rgba(167,139,250,.18));border:1px solid rgba(148,163,184,.22);margin-bottom:1rem;}
    .soft-card{background:linear-gradient(180deg, rgba(17,25,54,.96), rgba(10,17,38,.98));border:1px solid rgba(148,163,184,.24);border-radius:18px;padding:1rem;box-shadow:0 10px 35px rgba(0,0,0,.22);}
    .score-title{color:#bcd0ff;font-size:1.02rem;font-weight:800;}
    .score-value{font-size:3rem;font-weight:950;color:#fff;margin:.35rem 0;}
    .pill{display:inline-block;padding:.28rem .6rem;border-radius:999px;font-size:.82rem;font-weight:700;border:1px solid rgba(255,255,255,.12);margin-right:.35rem;}
    .pill-green{background:rgba(34,197,94,.16);color:#bbf7d0;}
    .pill-yellow{background:rgba(245,158,11,.16);color:#fde68a;}
    .pill-red{background:rgba(239,68,68,.16);color:#fecaca;}
    .pill-blue{background:rgba(56,189,248,.16);color:#bae6fd;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <div class="main-title">
      <div style="font-size:1.8rem;font-weight:900;">📈 {APP_NAME}</div>
      <div style="color:#98abd5;">Hard historical gates + sweet spots + improvement patterns + clustering + canary + lazy backtest.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

EASTERN = "America/New_York"
INVERSE_INDICATORS = {"$TRIN", "$VIX", "$CPCE", "VXX", "SPXS:SVOL"}
STATE_FEATURES = [
    "$BPSPX", "$BPSPX_%B", "$BPNYA", "$OEXA200R", "$SPXA50R", "$NYMO", "$NYSI", "$CPCE",
    "$NYHL", "$NYAD", "$SPXADP", "$TRIN", "$VIX", "RSP:SPY"
]
TREND_WINDOWS = [1, 2, 3, 5, 10]
CANARY_WEIGHTS = {
    "SPXS:SVOL": 0.20,
    "HYG:IEF": 0.18,
    "SMH:SPY": 0.16,
    "XLF:SPY": 0.12,
    "RSP:SPY": 0.12,
    "IWM:SPY": 0.10,
    "SPY:VXX": 0.07,
    "$VIX": 0.05,
}
OUTCOME_DEFS = {
    "bounce": {"horizon": 10, "ret": 0.03, "dd": -0.03, "type": "max"},
    "repair": {"horizon": 20, "ret": 0.04, "dd": -0.05, "type": "end"},
    "regime": {"horizon": 60, "ret": 0.08, "dd": -0.08, "type": "end"},
    "fall": {"horizon": 10, "ret": -0.03, "dd": 0.03, "type": "min_end"},
}

@dataclass
class ClusterArtifacts:
    scaler_mean: List[float]
    scaler_scale: List[float]
    features: List[str]
    centroids: List[List[float]]
    cluster_names: Dict[str, str]
    silhouette: float = 0.0


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


def signal_css(signal: str) -> str:
    return {"LONG": "pill-green", "SHORT": "pill-red"}.get(signal, "pill-yellow")


def detect_session_phase() -> str:
    ts = pd.Timestamp.now(tz=EASTERN)
    if ts.weekday() >= 5:
        return "Weekend"
    t = dt_time(ts.hour, ts.minute)
    if t < dt_time(9, 30):
        return "Pre-Market"
    if t < dt_time(11, 0):
        return "Opening"
    if t < dt_time(14, 30):
        return "Midday"
    if t < dt_time(16, 0):
        return "Late-Day"
    if t < dt_time(18, 0):
        return "Post-Close"
    return "Official EOD"


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


def stoch_from_close(close: pd.Series, length: int = 14, smoothk: int = 3) -> pd.Series:
    lo = close.rolling(length).min()
    hi = close.rolling(length).max()
    denom = (hi - lo).replace(0, np.nan)
    k = 100 * (close - lo) / denom
    return k.rolling(smoothk).mean()


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
        if all(pd.isna(x) for x in nums[:4]):
            continue
        rows.append({"date": dt, "open": nums[0], "high": nums[1], "low": nums[2], "close": nums[3], "volume": nums[4] if len(nums) > 4 else np.nan})
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
        "rsp_spy": "RSP:SPY", "smh_spy": "SMH:SPY", "iwm_spy": "IWM:SPY", "xlf_spy": "XLF:SPY",
        "spxs_svol": "SPXS:SVOL", "hyg_ief": "HYG:IEF", "spy_vxx": "SPY:VXX",
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
        raise ValueError("Snapshot CSV must contain a close-like column")
    out = pd.DataFrame({"symbol": df["Symbol"].astype(str).str.strip(), "close": pd.to_numeric(df[close_col], errors="coerce")})
    if "Daily PctChange" in df.columns:
        out["pct_change"] = pd.to_numeric(df["Daily PctChange"], errors="coerce")
    return out.dropna(subset=["symbol"]).reset_index(drop=True)


def add_indicator_features(hist: pd.DataFrame) -> pd.DataFrame:
    out = []
    for sym, g in hist.groupby("symbol", sort=False):
        g = g.sort_values("date").copy()
        close = g["close"]
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


def compute_future_metrics(price: pd.Series, horizon: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    vals = price.to_numpy(dtype=float)
    n = len(vals)
    idx = price.index
    end_ret = np.full(n, np.nan)
    max_gain = np.full(n, np.nan)
    max_dd = np.full(n, np.nan)
    valid = n - horizon
    if valid <= 0:
        return pd.Series(end_ret, index=idx), pd.Series(max_gain, index=idx), pd.Series(max_dd, index=idx)
    for i in range(valid):
        win = vals[i+1:i+horizon+1]
        if len(win) != horizon or np.isnan(vals[i]):
            continue
        rets = win / vals[i] - 1
        end_ret[i] = rets[-1]
        max_gain[i] = np.nanmax(rets)
        max_dd[i] = np.nanmin(rets)
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


def build_wide_features(feat: pd.DataFrame) -> pd.DataFrame:
    piv_close = feat.pivot(index="date", columns="symbol", values="close")
    piv_b = feat.pivot(index="date", columns="symbol", values="pct_b20")
    piv_rsi = feat.pivot(index="date", columns="symbol", values="rsi14")
    wide = pd.DataFrame(index=piv_close.index)
    for sym in piv_close.columns:
        wide[sym] = piv_close[sym]
        wide[f"{sym}_%B"] = piv_b[sym] if sym in piv_b.columns else np.nan
        wide[f"{sym}_RSI14"] = piv_rsi[sym] if sym in piv_rsi.columns else np.nan
        for w in TREND_WINDOWS:
            wide[f"{sym}_d{w}"] = piv_close[sym].diff(w)
            wide[f"{sym}_%B_d{w}"] = wide[f"{sym}_%B"].diff(w)
            wide[f"{sym}_RSI14_d{w}"] = wide[f"{sym}_RSI14"].diff(w)
    return wide.sort_index()


def choose_features(base: pd.DataFrame) -> List[str]:
    feats = []
    for feat in STATE_FEATURES:
        if feat in base.columns:
            feats.append(feat)
        for w in TREND_WINDOWS:
            for suffix in [f"_d{w}", f"_%B_d{w}", f"_RSI14_d{w}"]:
                col = f"{feat}{suffix}" if suffix.startswith("_") else f"{feat}{suffix}"
                if col in base.columns:
                    feats.append(col)
    return [c for c in feats if base[c].notna().sum() >= 80]


def direction_hints(feature: str, state: str) -> List[str]:
    if state == "fall":
        if any(k in feature for k in ["$TRIN", "$VIX", "$CPCE", "SPXS:SVOL"]):
            return ["gte", "lte"]
        return ["lte", "gte"]
    if any(k in feature for k in ["$TRIN", "$VIX", "$CPCE", "SPXS:SVOL"]):
        return ["lte", "gte"] if state in {"repair", "regime"} else ["gte", "lte"]
    if feature.endswith(tuple(f"_d{w}" for w in TREND_WINDOWS)) or "%B_d" in feature or "RSI14_d" in feature:
        return ["gte", "lte"] if state in {"bounce", "repair", "regime"} else ["lte", "gte"]
    return ["gte", "lte"] if state in {"repair", "regime"} else ["lte", "gte"]


def learn_single_gates(df: pd.DataFrame, features: List[str], label: str, min_support: int = 40, topn: int = 12) -> List[dict]:
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
                gates.append({"feature": feat, "direction": direction, "threshold": thr, "support": support, "hit_rate": hit, "base_rate": base, "lift": lift, "score": score})
    gates = sorted(gates, key=lambda x: (x["score"], x["lift"], x["support"]), reverse=True)
    top, used = [], set()
    for g in gates:
        if g["feature"] in used:
            continue
        used.add(g["feature"])
        top.append(g)
        if len(top) >= topn:
            break
    return top


def learn_combo_gates(df: pd.DataFrame, label: str, singles: List[dict], min_support: int = 25) -> List[dict]:
    y = df[f"{label}_success"].astype(float)
    base = float(y.mean())
    combos = []
    for i in range(len(singles)):
        for j in range(i + 1, min(len(singles), i + 7)):
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
            combos.append({"gates": [g1, g2], "support": support, "hit_rate": hit, "base_rate": base, "lift": hit / base if base > 0 else np.nan, "score": (hit - base) * math.sqrt(support)})
    return sorted(combos, key=lambda x: (x["score"], x["lift"], x["support"]), reverse=True)[:6]


def gate_pass(cur: float, gate: dict) -> bool:
    if pd.isna(cur):
        return False
    return cur >= gate["threshold"] if gate["direction"] == "gte" else cur <= gate["threshold"]


def summarize_bands(df: pd.DataFrame, label: str) -> Dict[str, Dict[str, float]]:
    hit = df[df[f"{label}_success"] == 1.0]
    res = {}
    for feat in STATE_FEATURES:
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
                cur, q25, med, q75 = -cur, -q75, -med, -q25
            sc = band_distance_score(cur, q25, med, q75)
            rows.append({"Outcome": label.title(), "Feature": feat, "Current": cur, "Median": med, "Q25": q25, "Q75": q75, "BandScore": sc})
            if pd.notna(sc):
                vals.append(sc)
        totals[label] = 100 * np.mean(vals) if vals else np.nan
    return pd.DataFrame(rows), totals


def build_clusters(base: pd.DataFrame, features: List[str], n_clusters: int = 6) -> Tuple[pd.DataFrame, ClusterArtifacts]:
    feat_df = base[features].apply(pd.to_numeric, errors="coerce")
    valid = feat_df.dropna()
    n_clusters = max(3, min(n_clusters, max(3, len(valid) // 40)))
    scaler = StandardScaler()
    X = scaler.fit_transform(valid)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20, max_iter=300)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else 0.0
    stats_rows = []
    for cl in sorted(np.unique(labels)):
        mask = labels == cl
        sub = valid.loc[mask]
        row = {"cluster": int(cl), "samples": int(mask.sum())}
        for k in ["bounce", "repair", "regime", "fall"]:
            row[f"{k}_rate"] = float(base.loc[sub.index, f"{k}_success"].mean())
        for feat in features:
            row[f"{feat}_median"] = float(sub[feat].median())
        stats_rows.append(row)
    stats_df = pd.DataFrame(stats_rows).sort_values("cluster").reset_index(drop=True)
    names = {}
    for idx, row in stats_df.iterrows():
        rates = {k: row[f"{k}_rate"] for k in ["bounce", "repair", "regime", "fall"]}
        best = max(rates, key=rates.get)
        if best == "fall" and rates[best] > 0.30:
            names[idx] = "Deterioration cluster"
        elif best == "regime" and rates[best] > 0.30:
            names[idx] = "Durable regime"
        elif best == "repair" and rates[best] > 0.22:
            names[idx] = "Repair cluster"
        elif best == "bounce" and rates[best] > 0.35:
            names[idx] = "Bounce cluster"
        elif row.get("$BPSPX_%B_median", 1.0) < 0.15:
            names[idx] = "Capitulation / washout"
        else:
            names[idx] = "Mixed / transitional"
    stats_df["cluster_name"] = stats_df["cluster"].map(names)
    artifacts = ClusterArtifacts(
        scaler_mean=scaler.mean_.tolist(), scaler_scale=scaler.scale_.tolist(), features=features,
        centroids=km.cluster_centers_.tolist(), cluster_names={str(k): v for k, v in names.items()}, silhouette=float(sil)
    )
    return stats_df, artifacts


def predict_cluster(snapshot: Dict[str, float], artifacts: ClusterArtifacts) -> Tuple[Optional[int], Optional[str], Optional[float]]:
    vals = []
    for feat in artifacts.features:
        v = safe_float(snapshot.get(feat, np.nan))
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


def build_canary_from_history(daily_feat: pd.DataFrame) -> pd.DataFrame:
    piv = daily_feat.pivot(index="date", columns="symbol", values="close")
    score_map = {}
    for sym, wt in CANARY_WEIGHTS.items():
        if sym not in piv.columns:
            continue
        close = piv[sym].dropna()
        if close.shape[0] < 220:
            continue
        mh = macd_hist(close)
        tsi = close.diff().ewm(span=20).mean() / close.diff().abs().ewm(span=20).mean() * 100
        stoch = stoch_from_close(close, 14, 3)
        cci100 = (close - close.rolling(100).mean()) / (0.015 * (close - close.rolling(100).mean()).abs().rolling(100).mean().replace(0, np.nan))
        df = pd.concat([mh.rename("macdh"), tsi.rename("tsi"), stoch.rename("stoch"), cci100.rename("cci")], axis=1).dropna()
        if df.empty:
            continue
        bull = (df["macdh"] > 0) & (df["tsi"] > 0) & (df["stoch"] > 50) & (df["cci"] > 0)
        bear = (df["macdh"] < 0) & (df["tsi"] < 0) & (df["stoch"] < 50) & (df["cci"] < 0)
        s = pd.Series(0.0, index=df.index)
        s[bull] = 1.0
        s[bear] = -1.0
        if sym in INVERSE_INDICATORS:
            s = -s
        score_map[sym] = s
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


def compute_recovery_momentum(base: pd.DataFrame, latest_idx) -> Dict[str, Any]:
    comps = {}
    for feat in ["$NYSI", "$NYMO", "$BPSPX_%B", "$SPXA50R", "$NYHL", "$TRIN", "$CPCE", "$VIX", "RSP:SPY"]:
        if feat not in base.columns:
            comps[feat] = np.nan
            continue
        s = base[feat].loc[:latest_idx].dropna()
        if len(s) < 6:
            comps[feat] = np.nan
            continue
        y = s.iloc[-6:].to_numpy(dtype=float)
        x = np.arange(len(y))
        xm, ym = np.mean(x), np.mean(y)
        denom = np.sum((x - xm) ** 2)
        slope = 0.0 if denom == 0 else np.sum((x - xm) * (y - ym)) / denom
        comps[feat] = slope / max(np.nanmean(np.abs(y)), 1e-6)
        if feat in INVERSE_INDICATORS and pd.notna(comps[feat]):
            comps[feat] = -comps[feat]
    vals = [np.tanh(v * 3.0) for v in comps.values() if pd.notna(v)]
    score = np.nan if not vals else float(np.clip(np.mean(vals) * 50 + 50, 0, 100))
    return {"recovery_score": score, "components": comps}


def compute_proxy(snapshot: Dict[str, float], prev_snapshot: Dict[str, float]) -> Dict[str, Any]:
    nyad = safe_float(snapshot.get("$NYAD", np.nan))
    spxadp = safe_float(snapshot.get("$SPXADP", np.nan))
    prev_nyad = safe_float(prev_snapshot.get("$NYAD", np.nan))
    prev_spxadp = safe_float(prev_snapshot.get("$SPXADP", np.nan))
    cur_raw = 0.6 * (0 if pd.isna(nyad) else nyad) + 0.4 * (0 if pd.isna(spxadp) else spxadp)
    prev_raw = 0.6 * (0 if pd.isna(prev_nyad) else prev_nyad) + 0.4 * (0 if pd.isna(prev_spxadp) else prev_spxadp)
    cur = 100 * np.tanh(cur_raw / 1600.0)
    prev = 100 * np.tanh(prev_raw / 1600.0)
    return {
        "nymo": float(cur),
        "nymo_delta": float(cur - prev),
        "nysi": float(np.clip(cur + 0.7 * safe_float(snapshot.get("$NYSI", 0)), -1000, 1000)),
        "state": "Deep washout" if cur <= -70 else "Negative but repairing" if cur <= -20 else "Neutral / crossing" if cur <= 20 else "Positive thrust",
    }


def evaluate_state(snapshot: Dict[str, float], state_model: Dict[str, Any]) -> Dict[str, Any]:
    singles = []
    for g in state_model.get("singles", []):
        cur = safe_float(snapshot.get(g["feature"], np.nan))
        singles.append({**g, "current": cur, "passed": gate_pass(cur, g)})
    combos = []
    for combo in state_model.get("combos", []):
        passed = all(gate_pass(safe_float(snapshot.get(g["feature"], np.nan)), g) for g in combo["gates"])
        combos.append({**combo, "passed": passed})
    pass_frac = float(np.mean([x["passed"] for x in singles])) if singles else 0.0
    base_rate = state_model.get("base_rate", np.nan)
    passed_hits = [x["hit_rate"] for x in singles if x["passed"]]
    passed_combo_hits = [x["hit_rate"] for x in combos if x["passed"]]
    prob = base_rate
    if passed_hits:
        prob = 0.55 * np.mean(passed_hits) + 0.25 * (np.mean(passed_combo_hits) if passed_combo_hits else base_rate) + 0.20 * base_rate
        prob = float(np.clip(prob * (0.65 + 0.35 * pass_frac), 0, 1)) if pd.notna(prob) else np.nan
    return {"prob": prob, "pass_frac": pass_frac, "base_rate": base_rate, "passed_singles": [x for x in singles if x["passed"]], "passed_combos": [x for x in combos if x["passed"]]}


def classify_signal(state_scores: Dict[str, Any], band_totals: Dict[str, float], canary: Dict[str, Any], cluster_name: Optional[str], recovery_score: float) -> Dict[str, Any]:
    bounce, repair, regime, fall = [state_scores[k] for k in ["bounce", "repair", "regime", "fall"]]
    long_flag = (((bounce["pass_frac"] >= 0.55 and bounce["prob"] >= max(0.40, bounce["base_rate"] + 0.08)) or (repair["pass_frac"] >= 0.50 and repair["prob"] >= max(0.30, repair["base_rate"] + 0.06))) and recovery_score >= 45 and canary["label"] != "Risk-Off") or ((regime["pass_frac"] >= 0.50 and regime["prob"] >= max(0.30, regime["base_rate"] + 0.05)) and canary["label"] != "Risk-Off")
    short_flag = (fall["pass_frac"] >= 0.50 and fall["prob"] >= max(0.25, fall["base_rate"] + 0.05) and canary["label"] != "Risk-On" and recovery_score < 40)
    signal = "LONG" if long_flag else "SHORT" if short_flag else "HOLD"
    reasons = []
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
        reasons.append(f"Canary: {canary['label']}")
    if cluster_name:
        reasons.append(f"Cluster: {cluster_name}")
    return {"signal": signal, "reasons": reasons, "bounce_prob": bounce["prob"], "repair_prob": repair["prob"], "regime_prob": regime["prob"], "fall_prob": fall["prob"]}


def calculate_risk_metrics(equity_curve: pd.Series, rf: float = 0.02) -> Dict[str, float]:
    returns = equity_curve.pct_change().dropna()
    if len(returns) < 20:
        return {k: np.nan for k in ["sharpe", "sortino", "calmar", "max_dd", "total_return", "cagr"]}
    excess = returns - rf / 252
    sharpe = (excess.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    downside = returns[returns < 0]
    sortino = (excess.mean() / downside.std()) * np.sqrt(252) if len(downside) > 0 and downside.std() > 0 else 0
    peak = equity_curve.cummax()
    dd = equity_curve / peak - 1
    max_dd = dd.min()
    calmar = (returns.mean() * 252) / abs(max_dd) if max_dd not in (0, np.nan) and pd.notna(max_dd) else 0
    years = len(equity_curve) / 252
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1 if years > 0 else 0
    return {"sharpe": sharpe, "sortino": sortino, "calmar": calmar, "max_dd": max_dd, "total_return": equity_curve.iloc[-1] / equity_curve.iloc[0] - 1, "cagr": cagr}


def build_model_from_history(daily: pd.DataFrame, weekly: pd.DataFrame) -> Dict[str, Any]:
    daily_feat = add_indicator_features(daily)
    weekly_feat = add_indicator_features(weekly) if not weekly.empty else weekly.copy()
    piv = daily_feat.pivot(index="date", columns="symbol", values="close")
    if "RSP" not in piv.columns:
        raise ValueError("RSP daily history is required")
    outcomes = build_outcomes(piv["RSP"].dropna())
    wide = build_wide_features(daily_feat)
    base = wide.join(outcomes, how="inner").dropna(subset=["bounce_success", "repair_success", "regime_success", "fall_success"])
    features = choose_features(base)
    learned = {"states": {}, "bands": {}, "meta": {"rows": int(len(base)), "feature_count": len(features)}}
    for state in ["bounce", "repair", "regime", "fall"]:
        singles = learn_single_gates(base, features, state)
        combos = learn_combo_gates(base, state, singles)
        learned["states"][state] = {"singles": singles, "combos": combos, "base_rate": float(base[f"{state}_success"].mean())}
        if state != "fall":
            learned["bands"][state] = summarize_bands(base, state)
    cluster_base = base.dropna(subset=[f for f in STATE_FEATURES if f in base.columns]).copy()
    cluster_stats, cluster_artifacts = build_clusters(cluster_base, [f for f in STATE_FEATURES if f in base.columns], n_clusters=6)
    learned["clusters"] = {"scaler_mean": cluster_artifacts.scaler_mean, "scaler_scale": cluster_artifacts.scaler_scale, "features": cluster_artifacts.features, "centroids": cluster_artifacts.centroids, "names": cluster_artifacts.cluster_names, "silhouette": cluster_artifacts.silhouette}
    canary_hist = build_canary_from_history(daily_feat)
    learned["canary_hist"] = canary_hist.reset_index().to_dict(orient="records") if not canary_hist.empty else []
    daily_feat.to_parquet(HIST_DAILY_PATH, index=False)
    if not weekly_feat.empty:
        weekly_feat.to_parquet(HIST_WEEKLY_PATH, index=False)
    save_json(MODEL_PATH, learned)
    return learned


def load_model() -> Optional[Dict[str, Any]]:
    if not MODEL_PATH.exists() or not HIST_DAILY_PATH.exists():
        return None
    return load_json(MODEL_PATH, None)


@st.cache_data(show_spinner=False)
def cached_parse_history(file_bytes: bytes) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return parse_stockcharts_zip(file_bytes)


@st.cache_data(show_spinner=False)
def cached_build_model(file_bytes: bytes) -> Dict[str, Any]:
    daily, weekly = parse_stockcharts_zip(file_bytes)
    return build_model_from_history(daily, weekly)


def build_snapshot_from_history_and_upload(daily_feat: pd.DataFrame, snap_df: Optional[pd.DataFrame]) -> Tuple[Dict[str, float], Dict[str, float], pd.Timestamp, pd.DataFrame]:
    latest_date = pd.to_datetime(daily_feat["date"]).max()
    daily_mod = daily_feat.copy()
    if snap_df is not None and not snap_df.empty:
        close_map = dict(zip(snap_df["symbol"].astype(str), snap_df["close"].astype(float)))
        mask = daily_mod["date"] == latest_date
        for sym, close in close_map.items():
            sel = mask & (daily_mod["symbol"] == sym)
            if sel.any():
                daily_mod.loc[sel, "close"] = close
                daily_mod.loc[sel, "high"] = np.maximum(daily_mod.loc[sel, "high"], close)
                daily_mod.loc[sel, "low"] = np.minimum(daily_mod.loc[sel, "low"], close)
        daily_mod = add_indicator_features(daily_mod)
    wide = build_wide_features(daily_mod)
    latest_date = wide.index.max()
    prior_date = wide.index[wide.index < latest_date].max() if (wide.index < latest_date).any() else latest_date
    snapshot = {c: safe_float(wide.loc[latest_date, c]) for c in wide.columns}
    prev_snapshot = {c: safe_float(wide.loc[prior_date, c]) for c in wide.columns}
    return snapshot, prev_snapshot, latest_date, wide


def render_score_card(title: str, score: float):
    pct = 0 if pd.isna(score) else max(0, min(100, score))
    fill = "fill-green" if pct >= 70 else "fill-yellow" if pct >= 40 else "fill-red"
    st.markdown(f'<div class="soft-card"><div class="score-title">{title}</div><div class="score-value">{fmt_num(pct,0)}%</div><div class="score-bar"><div class="score-fill {fill}" style="width:{pct:.1f}%"></div></div></div>', unsafe_allow_html=True)


def run_backtest_lazy(daily_feat: pd.DataFrame, model: Dict[str, Any], use_canary: bool, canary_thr: float, conf_thr: float, switch_cost_bps: float) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    wide = build_wide_features(daily_feat)
    piv = daily_feat.pivot(index="date", columns="symbol", values="close")
    canary_hist = pd.DataFrame(model.get("canary_hist", []))
    if not canary_hist.empty and "date" in canary_hist.columns:
        canary_hist["date"] = pd.to_datetime(canary_hist["date"])
        canary_hist = canary_hist.set_index("date")
    rows = []
    cl_art = model.get("clusters")
    artifacts = None
    if cl_art:
        artifacts = ClusterArtifacts(cl_art["scaler_mean"], cl_art["scaler_scale"], cl_art["features"], cl_art["centroids"], cl_art["names"], cl_art.get("silhouette", 0.0))
    dates = list(wide.index)
    for i in range(1, len(dates)):
        dt = dates[i]
        prev_dt = dates[i-1]
        snapshot = {c: safe_float(wide.loc[dt, c]) for c in wide.columns}
        prev_snapshot = {c: safe_float(wide.loc[prev_dt, c]) for c in wide.columns}
        proxy = compute_proxy(snapshot, prev_snapshot)
        state_scores = {state: evaluate_state(snapshot, model["states"][state]) for state in ["bounce", "repair", "regime", "fall"]}
        _, band_totals = score_bands(snapshot, model.get("bands", {}))
        recovery = compute_recovery_momentum(wide, dt)
        cluster_name = None
        if artifacts is not None:
            _, cluster_name, _ = predict_cluster(snapshot, artifacts)
        canary = {"label": "Neutral", "comp": 0.0, "conf": 0.0}
        if not canary_hist.empty and dt in canary_hist.index:
            comp = safe_float(canary_hist.loc[dt, "canary_comp"])
            conf = safe_float(canary_hist.loc[dt, "canary_conf"])
            canary = {"label": "Risk-On" if comp > 0.05 else "Risk-Off" if comp < -0.05 else "Neutral", "comp": comp, "conf": conf}
        signal = classify_signal(state_scores, band_totals, canary, cluster_name, recovery["recovery_score"])
        rows.append({"date": dt, "signal": signal["signal"], "rsp_close": safe_float(piv.loc[dt, "RSP"]) if "RSP" in piv.columns else np.nan, "spy_close": safe_float(piv.loc[dt, "SPY"]) if "SPY" in piv.columns else np.nan, "canary_comp": canary["comp"], "canary_conf": canary["conf"]})
    bt = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if bt.empty:
        return bt, {}
    bt["pos_long_hold"] = bt["signal"].map({"LONG": 1, "HOLD": 0, "SHORT": 0}).fillna(0)
    bt["pos_long_short"] = bt["signal"].map({"LONG": 1, "HOLD": 0, "SHORT": -1}).fillna(0)
    if use_canary:
        gate = (bt["canary_comp"].fillna(-1) > canary_thr) & (bt["canary_conf"].fillna(0) >= conf_thr)
        bt.loc[~gate, "pos_long_hold"] = 0
        bt.loc[~gate & (bt["pos_long_short"] > 0), "pos_long_short"] = 0
    bt["ret_rsp"] = bt["rsp_close"].pct_change().fillna(0)
    bt["ret_spy"] = bt["spy_close"].pct_change().fillna(0) if bt["spy_close"].notna().any() else np.nan
    cost = switch_cost_bps / 10000.0
    for name, poscol in [("lh", "pos_long_hold"), ("ls", "pos_long_short")]:
        sig = bt[poscol].shift(1).fillna(0)
        turnover = sig.diff().abs().fillna(0)
        bt[f"ret_{name}"] = sig * bt["ret_rsp"] - turnover * cost
        bt[f"equity_{name}"] = (1 + bt[f"ret_{name}"]).cumprod()
    bt["equity_rsp_bh"] = (1 + bt["ret_rsp"]).cumprod()
    if bt["ret_spy"].notna().any():
        bt["equity_spy_bh"] = (1 + bt["ret_spy"].fillna(0)).cumprod()
    metrics = {
        "Long/Hold": calculate_risk_metrics(bt["equity_lh"]),
        "Long/Short": calculate_risk_metrics(bt["equity_ls"]),
        "RSP BuyHold": calculate_risk_metrics(bt["equity_rsp_bh"]),
    }
    if "equity_spy_bh" in bt.columns:
        metrics["SPY BuyHold"] = calculate_risk_metrics(bt["equity_spy_bh"])
    return bt, metrics


def main():
    with st.sidebar:
        st.header("⚙️ Configuration")
        phase = detect_session_phase()
        st.write(f"Session: {phase}")
        use_proxy = st.toggle("Use proxy NYMO/NYSI intraday", value=(phase != "Official EOD"))
        force_rebuild = st.toggle("Force rebuild model", value=False)
        st.subheader("Canary")
        use_canary = st.toggle("Enable canary filter", value=True)
        canary_thr = st.slider("Canary threshold", -0.20, 0.30, 0.05, 0.01)
        conf_thr = st.slider("Canary confidence threshold", 0, 100, 55)
        switch_cost = st.slider("Backtest switch cost (bps)", 0.0, 25.0, 5.0, 0.5)
        hist_upload = st.file_uploader("Historical ZIP", type=["zip"])
        snap_upload = st.file_uploader("Daily snapshot CSV", type=["csv"])
        if st.button("Reset saved model", width='stretch'):
            for p in [HIST_DAILY_PATH, HIST_WEEKLY_PATH, MODEL_PATH, UPLOAD_HISTORY_PATH]:
                if p.exists():
                    p.unlink()
            st.success("Model reset")
            st.rerun()

    model = None
    if hist_upload is not None and (force_rebuild or not MODEL_PATH.exists()):
        with st.spinner("Building historical model..."):
            model = cached_build_model(hist_upload.read())
        st.sidebar.success("Historical model built and cached")
    if model is None:
        model = load_model()
    if model is None:
        st.info("Upload historical ZIP once to build the model.")
        return
    daily_feat = pd.read_parquet(HIST_DAILY_PATH)
    snap_df = parse_snapshot_csv(snap_upload.read()) if snap_upload is not None else None
    snapshot, prev_snapshot, latest_date, wide = build_snapshot_from_history_and_upload(daily_feat, snap_df)
    proxy = compute_proxy(snapshot, prev_snapshot)
    state_scores = {state: evaluate_state(snapshot, model["states"][state]) for state in ["bounce", "repair", "regime", "fall"]}
    band_df, band_totals = score_bands(snapshot, model.get("bands", {}))
    recovery = compute_recovery_momentum(wide, latest_date)
    canary_hist = pd.DataFrame(model.get("canary_hist", []))
    if not canary_hist.empty and "date" in canary_hist.columns:
        canary_hist["date"] = pd.to_datetime(canary_hist["date"])
        canary_hist = canary_hist.set_index("date")
    comp = conf = 0.0
    if not canary_hist.empty and latest_date in canary_hist.index:
        comp = safe_float(canary_hist.loc[latest_date, "canary_comp"])
        conf = safe_float(canary_hist.loc[latest_date, "canary_conf"])
    canary = {"label": "Risk-On" if comp > 0.05 else "Risk-Off" if comp < -0.05 else "Neutral", "comp": comp, "conf": conf}
    cluster_name = None
    if model.get("clusters"):
        cl = model["clusters"]
        artifacts = ClusterArtifacts(cl["scaler_mean"], cl["scaler_scale"], cl["features"], cl["centroids"], cl["names"], cl.get("silhouette", 0.0))
        _, cluster_name, cluster_conf = predict_cluster(snapshot, artifacts)
    signal = classify_signal(state_scores, band_totals, canary, cluster_name, recovery["recovery_score"])

    st.markdown(f"<div class='soft-card'><div class='score-title'>Daily Verdict</div><div class='score-value'>{signal['signal']}</div><span class='pill {signal_css(signal['signal'])}'>{signal['signal']}</span><span class='pill pill-blue'>{canary['label']}</span><span class='pill pill-yellow'>{phase}</span><div style='margin-top:.6rem'>{' | '.join(signal['reasons'])}</div></div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: render_score_card("Bounce", 100 * safe_float(signal["bounce_prob"]))
    with c2: render_score_card("Repair", 100 * safe_float(signal["repair_prob"]))
    with c3: render_score_card("Regime", 100 * safe_float(signal["regime_prob"]))
    with c4: render_score_card("Fall", 100 * safe_float(signal["fall_prob"]))

    tab1, tab2, tab3, tab4 = st.tabs(["Decision Dashboard", "Gate Diagnostics", "Band Alignment", "Backtest vs Buy & Hold"])
    with tab1:
        st.markdown("### Intraday Proxy Context")
        st.write({"Proxy NYMO": fmt_num(proxy['nymo']), "Δ NYMO": fmt_num(proxy['nymo_delta']), "Proxy NYSI": fmt_num(proxy['nysi']), "State": proxy['state'], "Recovery": fmt_num(recovery['recovery_score']), "Canary": canary['label'], "Cluster": cluster_name or 'n/a'})
        st.markdown("### Improvement Analysis")
        imp_rows = []
        for feat in ["$NYSI", "$NYMO", "$BPSPX_%B", "$SPXA50R", "$NYHL", "$TRIN", "$CPCE", "$VIX", "RSP:SPY"]:
            for w in [1,2,3,5,10]:
                col = f"{feat}_d{w}"
                if col in wide.columns:
                    imp_rows.append({"Feature": feat, "Window": w, "Delta": safe_float(snapshot.get(col, np.nan))})
        if imp_rows:
            st.dataframe(pd.DataFrame(imp_rows), width='stretch', hide_index=True)
    with tab2:
        rows = []
        for state in ["bounce", "repair", "regime", "fall"]:
            for r in state_scores[state]["passed_singles"][:8]:
                rows.append({"State": state, "Gate": f"{r['feature']} {'≥' if r['direction']=='gte' else '≤'} {fmt_num(r['threshold'],3)}", "Current": fmt_num(r['current'],3), "HitRate": fmt_num(r['hit_rate']*100,1), "Lift": fmt_num(r['lift'],2)})
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
    with tab3:
        st.dataframe(band_df, width='stretch', hide_index=True)
    with tab4:
        st.markdown("Backtest runs only when requested, to keep the main app fast.")
        if st.button("Run historical backtest", width='stretch'):
            with st.spinner("Running historical backtest..."):
                bt, metrics = run_backtest_lazy(daily_feat, model, use_canary, canary_thr, conf_thr, switch_cost)
            if bt.empty:
                st.warning("Backtest unavailable")
            else:
                st.dataframe(pd.DataFrame(metrics).T, width='stretch')
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=bt['date'], y=bt['equity_lh'], name='Strategy Long/Hold'))
                fig.add_trace(go.Scatter(x=bt['date'], y=bt['equity_ls'], name='Strategy Long/Short'))
                fig.add_trace(go.Scatter(x=bt['date'], y=bt['equity_rsp_bh'], name='RSP BuyHold'))
                if 'equity_spy_bh' in bt.columns:
                    fig.add_trace(go.Scatter(x=bt['date'], y=bt['equity_spy_bh'], name='SPY BuyHold'))
                fig.update_layout(height=520, title='Historical Backtest vs Buy & Hold', legend_orientation='h')
                st.plotly_chart(fig, width='stretch')
                st.dataframe(bt.tail(30), width='stretch', hide_index=True)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        st.error(f"Fatal error: {e}")
        st.code(str(e))
