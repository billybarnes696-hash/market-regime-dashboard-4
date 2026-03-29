
import io
import json
import math
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# -----------------------------
# Page / paths
# -----------------------------
st.set_page_config(page_title="Breadth Historical Gate Engine", layout="wide", page_icon="📈")

APP_DIR = Path("breadth_gate_store")
APP_DIR.mkdir(exist_ok=True)

DAILY_BASELINE_PATH = APP_DIR / "daily_baseline.parquet"
WEEKLY_BASELINE_PATH = APP_DIR / "weekly_baseline.parquet"
DAILY_FEATURES_PATH = APP_DIR / "daily_features.parquet"
WEEKLY_FEATURES_PATH = APP_DIR / "weekly_features.parquet"
MODEL_PATH = APP_DIR / "learned_model.json"
UPLOAD_HISTORY_PATH = APP_DIR / "upload_history.csv"
SNAPSHOT_DIR = APP_DIR / "snapshots"
SNAPSHOT_DIR.mkdir(exist_ok=True)

# -----------------------------
# Styling
# -----------------------------
CUSTOM_CSS = """
<style>
:root{
  --bg:#0b1020;
  --panel:#111936;
  --panel2:#162246;
  --text:#eaf0ff;
  --muted:#95a8d8;
  --green:#22c55e;
  --yellow:#f59e0b;
  --red:#ef4444;
  --blue:#38bdf8;
  --purple:#a78bfa;
}
.block-container{padding-top:1rem;padding-bottom:2rem;}
.soft-card{
  background:linear-gradient(180deg, rgba(17,25,54,.97), rgba(10,17,38,.99));
  border:1px solid rgba(148,163,184,.24);
  border-radius:18px;
  padding:1rem 1rem .95rem 1rem;
  box-shadow:0 10px 35px rgba(0,0,0,.22);
}
.main-title{
  padding:1rem 1.25rem;
  border-radius:18px;
  background:linear-gradient(135deg, rgba(56,189,248,.16), rgba(167,139,250,.16));
  border:1px solid rgba(148,163,184,.20);
  margin-bottom:1rem;
}
.metric-big{
  font-size:3rem;font-weight:950;line-height:1.0;color:white;margin:.15rem 0 .25rem 0;
}
.metric-label{
  color:#bcd0ff;font-size:1rem;font-weight:800;letter-spacing:.02em;
}
.pill{
  display:inline-block;padding:.3rem .6rem;border-radius:999px;font-size:.82rem;font-weight:800;
  border:1px solid rgba(255,255,255,.12);margin-right:.35rem;
}
.pill-green{background:rgba(34,197,94,.16); color:#bbf7d0;}
.pill-yellow{background:rgba(245,158,11,.16); color:#fde68a;}
.pill-red{background:rgba(239,68,68,.16); color:#fecaca;}
.pill-blue{background:rgba(56,189,248,.16); color:#bae6fd;}
.action-box{
  border-radius:16px;padding:.85rem 1rem;margin:.45rem 0;border:1px solid rgba(255,255,255,.10);
}
.action-long{background:rgba(34,197,94,.12);}
.action-short{background:rgba(239,68,68,.12);}
.action-hold{background:rgba(245,158,11,.12);}
.small-muted{color:#93a4cc;font-size:.88rem;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown("""
<div class="main-title">
  <div style="font-size:1.55rem;font-weight:900;">📈 Breadth Historical Gate Engine</div>
  <div class="small-muted">Hard bounce / repair / regime / fall gates learned from your uploaded StockCharts history. One-time historical upload, then daily snapshot verdicts: LONG / SHORT / HOLD.</div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Constants
# -----------------------------
FEATURE_LEVELS = ["close", "pct_b20", "rsi14", "cci20", "roc3", "slope3"]
FORWARD_DEFS = {
    "bounce": {"horizon": 10, "up": 0.03, "dd_floor": -0.03},
    "repair": {"horizon": 20, "ret": 0.04, "dd_floor": -0.05},
    "regime": {"horizon": 60, "ret": 0.08, "h20": 0.03, "dd_floor": -0.08},
    "fall": {"horizon": 10, "down": -0.03, "up_cap": 0.02},
}
STATE_COLORS = {"LONG": "green", "SHORT": "red", "HOLD": "yellow"}
SNAPSHOT_SYMBOL_MAP = {
    "$BPSPX":"$BPSPX", "$BPNYA":"$BPNYA", "$OEXA50R":"$OEXA50R", "$OEXA150R":"$OEXA150R", "$OEXA200R":"$OEXA200R",
    "$SPXA50R":"$SPXA50R", "$NYMO":"$NYMO", "$NYSI":"$NYSI", "$NYAD":"$NYAD", "$NYHL":"$NYHL",
    "$SPXADP":"$SPXADP", "$CPCE":"$CPCE", "$CPC":"$CPC", "$TRIN":"$TRIN", "$VIX":"VIX",
    "VXX":"VXX", "RSP":"RSP", "URSP":"URSP", "$SPX":"$SPX", "SPY":"SPY",
    "RSP:SPY":"RSP_SPY", "SMH:SPY":"SMH_SPY", "IWM:SPY":"IWM_SPY", "XLF:SPY":"XLF_SPY",
    "HYG:IEF":"HYG_IEF", "SPXS:SVOL":"SPXS_SVOL"
}
PRIMARY_DAILY_FEATURES = [
    "$BPSPX__pct_b20", "$BPSPX__close", "$BPNYA__close", "$OEXA200R__close", "$SPXA50R__close",
    "$NYMO__close", "$NYSI__close", "$NYAD__close", "$NYHL__close", "$SPXADP__close",
    "$CPCE__close", "$TRIN__close", "VIX__close", "VXX__close", "RSP_SPY__close",
    "SMH_SPY__close", "IWM_SPY__close", "XLF_SPY__close", "HYG_IEF__close", "SPXS_SVOL__close",
    "$BPSPX__cci20", "$SPXA50R__roc3", "$NYHL__slope3"
]
PRIMARY_WEEKLY_FEATURES = [
    "$NYSI__close", "$OEXA200R__close", "$SPXA50R__close", "$BPSPX__close", "$BPNYA__close",
    "$NYHL__close", "RSP_SPY__close", "SMH_SPY__close", "IWM_SPY__close"
]

# -----------------------------
# Helpers
# -----------------------------
def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def fmt_num(x, digits=2):
    if pd.isna(x):
        return "n/a"
    return f"{float(x):.{digits}f}"

def color_pill(signal: str) -> str:
    cls = {"LONG":"pill-green", "SHORT":"pill-red", "HOLD":"pill-yellow"}.get(signal, "pill-blue")
    return f'<span class="pill {cls}">{signal}</span>'

def save_json(path: Path, data: dict):
    path.write_text(json.dumps(data, indent=2))

def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default

def ensure_upload_history():
    if not UPLOAD_HISTORY_PATH.exists():
        pd.DataFrame(columns=["upload_ts", "snapshot_file", "verdict", "bounce_prob", "repair_prob", "regime_prob", "fall_prob"]).to_csv(UPLOAD_HISTORY_PATH, index=False)

def append_upload_history(row: Dict):
    ensure_upload_history()
    hist = pd.read_csv(UPLOAD_HISTORY_PATH)
    hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    hist.to_csv(UPLOAD_HISTORY_PATH, index=False)

def load_upload_history() -> pd.DataFrame:
    ensure_upload_history()
    try:
        return pd.read_csv(UPLOAD_HISTORY_PATH)
    except Exception:
        return pd.DataFrame()

def save_snapshot_file(df: pd.DataFrame) -> Path:
    path = SNAPSHOT_DIR / f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(path, index=False)
    return path

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1 / period, adjust=False).mean()
    ma_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)

def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    tp = (high + low + close) / 3.0
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))

def percent_b(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    denom = (upper - lower).replace(0, np.nan)
    return (series - lower) / denom

def roc(series: pd.Series, period: int = 3) -> pd.Series:
    return 100 * (series / series.shift(period) - 1)

def slope_n(series: pd.Series, n: int = 3) -> pd.Series:
    return series - series.shift(n)

def compute_proxy_nymo(snapshot: Dict[str, float], prev_snapshot: Dict[str, float]) -> Dict[str, float]:
    nyad = safe_float(snapshot.get("$NYAD__close", np.nan))
    spxadp = safe_float(snapshot.get("$SPXADP__close", np.nan))
    prev_nyad = safe_float(prev_snapshot.get("$NYAD__close", np.nan))
    prev_spxadp = safe_float(prev_snapshot.get("$SPXADP__close", np.nan))
    cur_raw = 0.6 * (0 if pd.isna(nyad) else nyad) + 0.4 * (0 if pd.isna(spxadp) else spxadp)
    prev_raw = 0.6 * (0 if pd.isna(prev_nyad) else prev_nyad) + 0.4 * (0 if pd.isna(prev_spxadp) else prev_spxadp)
    val = 100 * np.tanh(cur_raw / 1600.0)
    prev_val = 100 * np.tanh(prev_raw / 1600.0)
    delta = val - prev_val
    return {"proxy_nymo": val, "proxy_delta": delta, "proxy_raw": cur_raw}

def compute_proxy_nysi(snapshot: Dict[str, float], prev_snapshot: Dict[str, float]) -> Dict[str, float]:
    proxy = compute_proxy_nymo(snapshot, prev_snapshot)
    official_prev = safe_float(prev_snapshot.get("$NYSI__close", np.nan))
    if pd.isna(official_prev):
        official_prev = safe_float(snapshot.get("$NYSI__close", np.nan))
    val = official_prev + (proxy["proxy_nymo"] / 12.0)
    prev_val = official_prev + ((proxy["proxy_nymo"] - proxy["proxy_delta"]) / 12.0)
    return {"proxy_nysi": val, "proxy_delta": val - prev_val}

# -----------------------------
# Parsing
# -----------------------------
@st.cache_data(show_spinner=False)
def parse_stockcharts_zip(file_bytes: bytes) -> Tuple[pd.DataFrame, pd.DataFrame]:
    daily_records, weekly_records = [], []
    with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        for name in csv_names:
            raw = zf.read(name).decode("utf-8", errors="ignore").splitlines()
            if len(raw) < 3:
                continue
            first = raw[0].strip()
            sym = first.split(",")[0].strip()
            if sym == "":
                continue
            if sym.lower().startswith("stockcharts/"):
                sym = Path(sym).name
            timeframe = "weekly" if " w.csv" in name.lower() else "daily"
            rows = []
            for line in raw[2:]:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 6:
                    continue
                dt = pd.to_datetime(parts[0], format="%m/%d/%Y", errors="coerce")
                if pd.isna(dt):
                    continue
                rows.append({
                    "date": dt, "symbol": sym, "open": safe_float(parts[1]), "high": safe_float(parts[2]),
                    "low": safe_float(parts[3]), "close": safe_float(parts[4]), "volume": safe_float(parts[5])
                })
            if not rows:
                continue
            if timeframe == "daily":
                daily_records.extend(rows)
            else:
                weekly_records.extend(rows)
    daily = pd.DataFrame(daily_records).sort_values(["symbol", "date"]).reset_index(drop=True)
    weekly = pd.DataFrame(weekly_records).sort_values(["symbol", "date"]).reset_index(drop=True)
    if daily.empty:
        raise ValueError("Could not parse any daily CSV files from the historical zip.")
    return daily, weekly

@st.cache_data(show_spinner=False)
def parse_realtime_snapshot(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    if "Symbol" not in df.columns:
        raise ValueError("Snapshot file must contain a 'Symbol' column.")
    out = df.copy()
    out["Symbol"] = out["Symbol"].astype(str).str.strip()
    close_col = None
    for c in ["Close", "Last", "Price", "Current", "Value"]:
        if c in out.columns:
            close_col = c
            break
    if close_col is None:
        raise ValueError("Snapshot file must contain a close-like column such as 'Close'.")
    out["Close"] = pd.to_numeric(out[close_col], errors="coerce")
    pct_col = None
    for c in ["Daily PctChange", "Daily PctChange(1,Close)", "Daily PctChange(1,Daily Close)", "% Change", "PctChange", "Pct Change"]:
        if c in out.columns:
            pct_col = c
            break
    out["PctChange"] = pd.to_numeric(out[pct_col], errors="coerce") if pct_col else np.nan
    out = out[["Symbol", "Close", "PctChange"]].dropna(subset=["Close"])
    if out.empty:
        raise ValueError("Snapshot file parsed successfully, but no valid close values were found.")
    return out

# -----------------------------
# Feature engineering
# -----------------------------
@st.cache_data(show_spinner=False)
def add_indicator_features(hist: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for sym, g in hist.groupby("symbol", sort=False):
        g = g.sort_values("date").copy()
        g["rsi14"] = rsi(g["close"], 14)
        g["cci20"] = cci(g["high"], g["low"], g["close"], 20)
        g["pct_b20"] = percent_b(g["close"], 20, 2.0)
        g["roc3"] = roc(g["close"], 3)
        g["slope3"] = slope_n(g["close"], 3)
        frames.append(g)
    return pd.concat(frames, ignore_index=True)

def features_wide(hist_feat: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
    pieces = []
    for feat in feature_list:
        p = hist_feat.pivot(index="date", columns="symbol", values=feat)
        p.columns = [f"{c}__{feat}" for c in p.columns]
        pieces.append(p)
    wide = pd.concat(pieces, axis=1).sort_index()
    return wide

def recompute_latest_indicators_from_snapshot(hist_feat: pd.DataFrame, snapshot_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    rt_map = {str(r["Symbol"]).strip(): safe_float(r["Close"]) for _, r in snapshot_df.iterrows()}
    results = {}
    for snap_sym, close_val in rt_map.items():
        mapped = SNAPSHOT_SYMBOL_MAP.get(snap_sym, snap_sym)
        g = hist_feat[hist_feat["symbol"] == mapped].sort_values("date").copy()
        if g.empty:
            continue
        g.iloc[-1, g.columns.get_loc("close")] = close_val
        if len(g) > 1:
            g.iloc[-1, g.columns.get_loc("open")] = g.iloc[-2]["close"]
        g.iloc[-1, g.columns.get_loc("high")] = max(g.iloc[-1]["high"], close_val)
        g.iloc[-1, g.columns.get_loc("low")] = min(g.iloc[-1]["low"], close_val)
        g["rsi14"] = rsi(g["close"], 14)
        g["cci20"] = cci(g["high"], g["low"], g["close"], 20)
        g["pct_b20"] = percent_b(g["close"], 20, 2.0)
        g["roc3"] = roc(g["close"], 3)
        g["slope3"] = slope_n(g["close"], 3)
        last = g.iloc[-1]
        results[mapped] = {
            "close": safe_float(last["close"]), "pct_b20": safe_float(last["pct_b20"]), "rsi14": safe_float(last["rsi14"]),
            "cci20": safe_float(last["cci20"]), "roc3": safe_float(last["roc3"]), "slope3": safe_float(last["slope3"])
        }
    return results

def latest_snapshot_dict(hist_feat: pd.DataFrame) -> Dict[str, float]:
    snapshot = {}
    for sym, g in hist_feat.groupby("symbol", sort=False):
        row = g.sort_values("date").iloc[-1]
        for feat in FEATURE_LEVELS:
            snapshot[f"{sym}__{feat}"] = safe_float(row.get(feat))
    return snapshot

def prior_snapshot_dict(hist_feat: pd.DataFrame) -> Dict[str, float]:
    snapshot = {}
    for sym, g in hist_feat.groupby("symbol", sort=False):
        g = g.sort_values("date")
        if len(g) < 2:
            continue
        row = g.iloc[-2]
        for feat in FEATURE_LEVELS:
            snapshot[f"{sym}__{feat}"] = safe_float(row.get(feat))
    return snapshot

# -----------------------------
# Forward outcomes / gate learning
# -----------------------------
def build_forward_outcomes(daily_feat: pd.DataFrame) -> pd.DataFrame:
    rsp = daily_feat[daily_feat["symbol"] == "RSP"].sort_values("date").copy()
    if rsp.empty:
        raise ValueError("Historical zip must contain RSP daily history.")
    close = rsp["close"].astype(float)
    idx = rsp["date"].values

    def max_gain(window):
        return np.nanmax(window / window[0] - 1.0) if len(window) else np.nan

    def max_drawdown(window):
        return np.nanmin(window / window[0] - 1.0) if len(window) else np.nan

    def fwd_ret(window):
        return (window[-1] / window[0] - 1.0) if len(window) else np.nan

    h10_max = close.rolling(FORWARD_DEFS["bounce"]["horizon"] + 1).apply(max_gain, raw=True).shift(-FORWARD_DEFS["bounce"]["horizon"])
    h10_dd = close.rolling(FORWARD_DEFS["bounce"]["horizon"] + 1).apply(max_drawdown, raw=True).shift(-FORWARD_DEFS["bounce"]["horizon"])
    h20_ret = close.rolling(FORWARD_DEFS["repair"]["horizon"] + 1).apply(fwd_ret, raw=True).shift(-FORWARD_DEFS["repair"]["horizon"])
    h20_dd = close.rolling(FORWARD_DEFS["repair"]["horizon"] + 1).apply(max_drawdown, raw=True).shift(-FORWARD_DEFS["repair"]["horizon"])
    h60_ret = close.rolling(FORWARD_DEFS["regime"]["horizon"] + 1).apply(fwd_ret, raw=True).shift(-FORWARD_DEFS["regime"]["horizon"])
    h60_dd = close.rolling(FORWARD_DEFS["regime"]["horizon"] + 1).apply(max_drawdown, raw=True).shift(-FORWARD_DEFS["regime"]["horizon"])
    h20_recheck = close.rolling(21).apply(fwd_ret, raw=True).shift(-20)
    h10_min = close.rolling(FORWARD_DEFS["fall"]["horizon"] + 1).apply(max_drawdown, raw=True).shift(-FORWARD_DEFS["fall"]["horizon"])
    h10_up = close.rolling(FORWARD_DEFS["fall"]["horizon"] + 1).apply(max_gain, raw=True).shift(-FORWARD_DEFS["fall"]["horizon"])

    out = pd.DataFrame({
        "date": rsp["date"].values,
        "bounce": (h10_max >= FORWARD_DEFS["bounce"]["up"]) & (h10_dd >= FORWARD_DEFS["bounce"]["dd_floor"]),
        "repair": (h20_ret >= FORWARD_DEFS["repair"]["ret"]) & (h20_dd >= FORWARD_DEFS["repair"]["dd_floor"]),
        "regime": (h60_ret >= FORWARD_DEFS["regime"]["ret"]) & (h20_recheck >= FORWARD_DEFS["regime"]["h20"]) & (h60_dd >= FORWARD_DEFS["regime"]["dd_floor"]),
        "fall": (h10_min <= FORWARD_DEFS["fall"]["down"]) & (h10_up <= FORWARD_DEFS["fall"]["up_cap"]),
        "fwd10_max": h10_max,
        "fwd20_ret": h20_ret,
        "fwd60_ret": h60_ret
    }).dropna()
    return out

def pick_feature_columns(wide: pd.DataFrame, candidates: List[str]) -> List[str]:
    return [c for c in candidates if c in wide.columns]

def gate_direction_hint(feature: str, state: str) -> List[str]:
    # bias search toward sensible directions but still test both if ambiguous
    lower_feats = ["pct_b20", "cci20"]
    panic_high_feats = ["$TRIN__close", "$CPCE__close", "VIX__close", "VXX__close", "SPXS_SVOL__close"]
    positive_feats = ["$SPXA50R__close", "$BPSPX__close", "$BPNYA__close", "$OEXA200R__close", "$NYMO__close", "$NYSI__close", "$NYHL__close", "$NYAD__close", "$SPXADP__close", "RSP_SPY__close", "SMH_SPY__close", "IWM_SPY__close", "XLF_SPY__close", "HYG_IEF__close"]
    if state == "bounce":
        if any(x in feature for x in lower_feats):
            return ["gte", "lte"]
        if feature in panic_high_feats:
            return ["gte", "lte"]
        if feature in positive_feats:
            return ["lte", "gte"]
    if state in {"repair", "regime"}:
        return ["gte", "lte"]
    if state == "fall":
        if feature in panic_high_feats:
            return ["gte", "lte"]
        return ["lte", "gte"]
    return ["gte", "lte"]

def learn_single_gates(train_df: pd.DataFrame, feature_cols: List[str], state: str, min_support: int = 30) -> List[dict]:
    y = train_df[state].astype(int)
    base = float(y.mean()) if len(y) else np.nan
    gates = []
    for col in feature_cols:
        s = pd.to_numeric(train_df[col], errors="coerce")
        valid = s.notna() & y.notna()
        sv = s[valid]
        yv = y[valid]
        if len(sv) < max(min_support * 2, 80):
            continue
        q_vals = sorted(set([round(x, 6) for x in sv.quantile(np.linspace(0.15, 0.85, 15)).tolist() if pd.notna(x)]))
        for direction in gate_direction_hint(col, state):
            for thr in q_vals:
                mask = sv >= thr if direction == "gte" else sv <= thr
                support = int(mask.sum())
                if support < min_support:
                    continue
                hit = float(yv[mask].mean())
                lift = hit / base if base and base > 0 else np.nan
                pass_rate = support / len(sv)
                score = (hit - base) * math.sqrt(support) + 0.05 * pass_rate
                gates.append({
                    "feature": col, "direction": direction, "threshold": float(thr), "support": support,
                    "hit_rate": hit, "base_rate": base, "lift": lift, "score": score
                })
    gates = [g for g in gates if np.isfinite(g["score"]) and g["hit_rate"] > g["base_rate"]]
    gates = sorted(gates, key=lambda x: (x["score"], x["lift"], x["support"]), reverse=True)
    top = []
    used = set()
    for g in gates:
        if g["feature"] in used:
            continue
        top.append(g)
        used.add(g["feature"])
        if len(top) >= 8:
            break
    return top

def eval_gate(df: pd.DataFrame, gate: dict) -> pd.Series:
    s = pd.to_numeric(df[gate["feature"]], errors="coerce")
    return (s >= gate["threshold"]) if gate["direction"] == "gte" else (s <= gate["threshold"])

def learn_combo_gates(train_df: pd.DataFrame, state: str, singles: List[dict], min_support: int = 25) -> List[dict]:
    y = train_df[state].astype(int)
    base = float(y.mean())
    combos = []
    for i in range(len(singles)):
        for j in range(i + 1, len(singles)):
            g1, g2 = singles[i], singles[j]
            mask = eval_gate(train_df, g1) & eval_gate(train_df, g2)
            support = int(mask.sum())
            if support < min_support:
                continue
            hit = float(y[mask].mean())
            if hit <= base:
                continue
            lift = hit / base if base > 0 else np.nan
            score = (hit - base) * math.sqrt(support)
            combos.append({
                "gates": [g1, g2], "support": support, "hit_rate": hit, "base_rate": base, "lift": lift, "score": score
            })
    combos = sorted(combos, key=lambda x: (x["score"], x["lift"], x["support"]), reverse=True)
    return combos[:5]

def learn_gate_model(daily_wide: pd.DataFrame, weekly_wide: pd.DataFrame, outcomes: pd.DataFrame) -> dict:
    model_df = daily_wide.join(outcomes.set_index("date"), how="inner").dropna(subset=["bounce", "repair", "regime", "fall"])
    daily_features = pick_feature_columns(model_df, PRIMARY_DAILY_FEATURES)
    learned = {"states": {}, "meta": {"rows": int(len(model_df))}}
    for state in ["bounce", "repair", "regime", "fall"]:
        singles = learn_single_gates(model_df, daily_features, state)
        combos = learn_combo_gates(model_df, state, singles)
        learned["states"][state] = {"singles": singles, "combos": combos, "base_rate": float(model_df[state].mean())}
    # weekly regime overlay
    if weekly_wide is not None and not weekly_wide.empty:
        wk = weekly_wide.copy()
        regime_future = outcomes[["date", "regime"]].sort_values("date")
        wk = pd.merge_asof(wk.sort_index().reset_index().rename(columns={"index":"date"}), regime_future, on="date", direction="forward").dropna(subset=["regime"])
        weekly_features = pick_feature_columns(wk, PRIMARY_WEEKLY_FEATURES)
        singles = learn_single_gates(wk, weekly_features, "regime", min_support=10)
        combos = learn_combo_gates(wk, "regime", singles, min_support=8)
        learned["weekly_regime"] = {"singles": singles[:6], "combos": combos[:4], "base_rate": float(wk["regime"].mean())}
    else:
        learned["weekly_regime"] = {"singles": [], "combos": [], "base_rate": np.nan}
    return learned

def gate_to_text(g: dict) -> str:
    op = "≥" if g["direction"] == "gte" else "≤"
    return f"{g['feature']} {op} {g['threshold']:.3f}"

# -----------------------------
# Live scoring / verdict
# -----------------------------
def snapshot_to_feature_row(snapshot: Dict[str, float], feature_cols: List[str]) -> pd.DataFrame:
    return pd.DataFrame([{c: snapshot.get(c, np.nan) for c in feature_cols}])

def score_state(snapshot: Dict[str, float], state_model: dict) -> dict:
    single_results = []
    for g in state_model.get("singles", []):
        cur = safe_float(snapshot.get(g["feature"], np.nan))
        passed = (cur >= g["threshold"]) if g["direction"] == "gte" else (cur <= g["threshold"])
        single_results.append({
            "text": gate_to_text(g), "passed": bool(passed) if pd.notna(cur) else False, "current": cur,
            "hit_rate": g["hit_rate"], "lift": g["lift"], "feature": g["feature"], "direction": g["direction"], "threshold": g["threshold"]
        })
    combo_results = []
    for combo in state_model.get("combos", []):
        gate_passes = []
        for g in combo["gates"]:
            cur = safe_float(snapshot.get(g["feature"], np.nan))
            gate_passes.append((cur >= g["threshold"]) if g["direction"] == "gte" else (cur <= g["threshold"]))
        passed = all(bool(x) for x in gate_passes)
        combo_results.append({"text": " AND ".join(gate_to_text(g) for g in combo["gates"]), "passed": passed, "hit_rate": combo["hit_rate"], "lift": combo["lift"]})
    pass_frac = (sum(r["passed"] for r in single_results) / len(single_results)) if single_results else 0.0
    passed_hits = [r["hit_rate"] for r in single_results if r["passed"]]
    passed_lifts = [r["lift"] for r in single_results if r["passed"]]
    combo_hits = [r["hit_rate"] for r in combo_results if r["passed"]]
    base = state_model.get("base_rate", np.nan)
    prob = base
    if passed_hits:
        prob = 0.55 * np.mean(passed_hits) + 0.25 * (np.mean(combo_hits) if combo_hits else base) + 0.20 * base
    prob = float(np.clip(prob * (0.65 + 0.35 * pass_frac), 0, 1)) if pd.notna(prob) else np.nan
    return {
        "prob": prob, "pass_frac": pass_frac, "passed_singles": [r for r in single_results if r["passed"]],
        "failed_singles": [r for r in single_results if not r["passed"]], "passed_combos": [r for r in combo_results if r["passed"]],
        "single_results": single_results, "combo_results": combo_results, "base_rate": base
    }

def compute_canary_filter(snapshot: Dict[str, float], prev_snapshot: Dict[str, float]) -> dict:
    canary_features = ["RSP_SPY__close", "SMH_SPY__close", "IWM_SPY__close", "XLF_SPY__close", "HYG_IEF__close"]
    stress_feats = ["SPXS_SVOL__close", "VXX__close", "VIX__close"]
    score = 0
    reasons = []
    for f in canary_features:
        cur, prev = safe_float(snapshot.get(f, np.nan)), safe_float(prev_snapshot.get(f, np.nan))
        if pd.notna(cur) and pd.notna(prev) and cur > prev:
            score += 1
            reasons.append(f"{f} improving")
    for f in stress_feats:
        cur, prev = safe_float(snapshot.get(f, np.nan)), safe_float(prev_snapshot.get(f, np.nan))
        if pd.notna(cur) and pd.notna(prev) and cur < prev:
            score += 1
            reasons.append(f"{f} easing")
    label = "Risk-On" if score >= 5 else "Risk-Off" if score <= 2 else "Neutral"
    return {"score": score, "label": label, "reasons": reasons}

def compute_intraday_context(snapshot: Dict[str, float], prev_snapshot: Dict[str, float], use_proxy: bool) -> dict:
    p_nymo = compute_proxy_nymo(snapshot, prev_snapshot)
    p_nysi = compute_proxy_nysi(snapshot, prev_snapshot)
    official_nymo = safe_float(snapshot.get("$NYMO__close", np.nan))
    official_nysi = safe_float(snapshot.get("$NYSI__close", np.nan))
    eff_nymo = p_nymo["proxy_nymo"] if use_proxy or pd.isna(official_nymo) else official_nymo
    eff_nysi = p_nysi["proxy_nysi"] if use_proxy or pd.isna(official_nysi) else official_nysi
    return {
        "nymo": eff_nymo, "nymo_delta": p_nymo["proxy_delta"] if use_proxy or pd.isna(official_nymo) else official_nymo - safe_float(prev_snapshot.get("$NYMO__close", np.nan)),
        "nysi": eff_nysi, "nysi_delta": p_nysi["proxy_delta"] if use_proxy or pd.isna(official_nysi) else official_nysi - safe_float(prev_snapshot.get("$NYSI__close", np.nan)),
        "mode": "Proxy" if use_proxy or pd.isna(official_nymo) else "Official"
    }

def classify_signal(state_scores: dict, intraday: dict, canary: dict, weekly_score: Optional[dict]) -> dict:
    bounce = state_scores["bounce"]
    repair = state_scores["repair"]
    regime = state_scores["regime"]
    fall = state_scores["fall"]
    weekly_pass = weekly_score["pass_frac"] if weekly_score else 0.0

    signal = "HOLD"
    reasons = []
    gate_params = {}

    # hard directional logic
    long_flag = (
        (bounce["pass_frac"] >= 0.60 and bounce["prob"] >= max(0.45, bounce["base_rate"] + 0.08)) or
        (repair["pass_frac"] >= 0.60 and repair["prob"] >= max(0.35, repair["base_rate"] + 0.06)) or
        (regime["pass_frac"] >= 0.55 and regime["prob"] >= max(0.40, regime["base_rate"] + 0.05) and weekly_pass >= 0.40)
    )
    short_flag = (
        (fall["pass_frac"] >= 0.60 and fall["prob"] >= max(0.40, fall["base_rate"] + 0.08)) and
        (repair["pass_frac"] < 0.45) and (regime["pass_frac"] < 0.40)
    )

    if long_flag and canary["label"] != "Risk-Off":
        signal = "LONG"
    elif short_flag and canary["label"] != "Risk-On":
        signal = "SHORT"
    else:
        signal = "HOLD"

    def add_reasons(label, score_obj, n=3):
        for r in score_obj["passed_singles"][:n]:
            reasons.append(f"{label}: {r['feature']} {'≥' if r['direction']=='gte' else '≤'} {r['threshold']:.3f}")
            gate_params[r["feature"]] = snapshot_value_str(r["current"], r["threshold"], r["direction"])

    if signal == "LONG":
        if repair["pass_frac"] >= bounce["pass_frac"] and repair["pass_frac"] >= regime["pass_frac"]:
            add_reasons("Repair", repair)
        elif regime["pass_frac"] >= bounce["pass_frac"]:
            add_reasons("Regime", regime)
        else:
            add_reasons("Bounce", bounce)
        reasons.append(f"Canary: {canary['label']}")
        reasons.append(f"NYMO context: {intraday['mode']} {fmt_num(intraday['nymo'])}")
    elif signal == "SHORT":
        add_reasons("Fall", fall)
        reasons.append(f"Canary: {canary['label']}")
        reasons.append(f"NYMO context: {intraday['mode']} {fmt_num(intraday['nymo'])}")
    else:
        reasons.append("Mixed historical gates; no strong edge.")
        reasons.append(f"Bounce {bounce['pass_frac']:.0%} / Repair {repair['pass_frac']:.0%} / Regime {regime['pass_frac']:.0%} / Fall {fall['pass_frac']:.0%}")
        reasons.append(f"Canary: {canary['label']}")
    return {"signal": signal, "reasons": reasons[:6], "gate_params": gate_params}

def snapshot_value_str(current: float, threshold: float, direction: str) -> str:
    op = "≥" if direction == "gte" else "≤"
    return f"{fmt_num(current)} vs {op} {threshold:.3f}"

# -----------------------------
# Persistence model builder
# -----------------------------
def build_and_persist_model(hist_zip_bytes: bytes):
    daily, weekly = parse_stockcharts_zip(hist_zip_bytes)
    daily.to_parquet(DAILY_BASELINE_PATH, index=False)
    if not weekly.empty:
        weekly.to_parquet(WEEKLY_BASELINE_PATH, index=False)
    daily_feat = add_indicator_features(daily)
    daily_feat.to_parquet(DAILY_FEATURES_PATH, index=False)
    weekly_feat = pd.DataFrame()
    if not weekly.empty:
        weekly_feat = add_indicator_features(weekly)
        weekly_feat.to_parquet(WEEKLY_FEATURES_PATH, index=False)
    outcomes = build_forward_outcomes(daily_feat)
    daily_wide = features_wide(daily_feat, FEATURE_LEVELS)
    weekly_wide = features_wide(weekly_feat, FEATURE_LEVELS) if not weekly_feat.empty else pd.DataFrame()
    model = learn_gate_model(daily_wide, weekly_wide, outcomes)
    save_json(MODEL_PATH, model)
    return daily, weekly, daily_feat, weekly_feat, model

def load_persisted_artifacts():
    if not (DAILY_FEATURES_PATH.exists() and MODEL_PATH.exists()):
        return None, None, None, None, None
    daily = pd.read_parquet(DAILY_BASELINE_PATH) if DAILY_BASELINE_PATH.exists() else None
    weekly = pd.read_parquet(WEEKLY_BASELINE_PATH) if WEEKLY_BASELINE_PATH.exists() else pd.DataFrame()
    daily_feat = pd.read_parquet(DAILY_FEATURES_PATH)
    weekly_feat = pd.read_parquet(WEEKLY_FEATURES_PATH) if WEEKLY_FEATURES_PATH.exists() else pd.DataFrame()
    model = load_json(MODEL_PATH, {})
    return daily, weekly, daily_feat, weekly_feat, model

# -----------------------------
# UI sidebar / data loading
# -----------------------------
st.sidebar.header("Model Inputs")
use_proxy = st.sidebar.toggle("Use proxy NYMO/NYSI before official evening data", value=True)
force_rebuild = st.sidebar.toggle("Force rebuild historical model", value=False)
if st.sidebar.button("Reset saved historical model"):
    for p in [DAILY_BASELINE_PATH, WEEKLY_BASELINE_PATH, DAILY_FEATURES_PATH, WEEKLY_FEATURES_PATH, MODEL_PATH]:
        if p.exists():
            p.unlink()
    st.sidebar.success("Saved historical model cleared. Upload the historical zip again.")

hist_file = st.sidebar.file_uploader("One-time historical upload (.zip)", type=["zip"])
snap_file = st.sidebar.file_uploader("Daily snapshot upload (.csv)", type=["csv"])

daily, weekly, daily_feat, weekly_feat, model = load_persisted_artifacts()
if hist_file is not None and (force_rebuild or daily_feat is None):
    with st.spinner("Building historical gate model from uploaded zip..."):
        daily, weekly, daily_feat, weekly_feat, model = build_and_persist_model(hist_file.getvalue())
    st.success("Historical gate model built and saved. You will not need to upload the historical zip again unless you want to rebuild the model.")
elif daily_feat is not None and model:
    st.info("Using saved historical gate model. Historical upload is not required again unless you want to refresh the model.")

if daily_feat is None or not model:
    st.warning("Upload your historical StockCharts zip once to build the gate model.")
    st.stop()

# -----------------------------
# Current snapshot assembly
# -----------------------------
base_snapshot = latest_snapshot_dict(daily_feat)
prev_snapshot = prior_snapshot_dict(daily_feat)

if snap_file is not None:
    snapshot_df = parse_realtime_snapshot(snap_file.getvalue())
    snap_path = save_snapshot_file(snapshot_df)
    overrides = recompute_latest_indicators_from_snapshot(daily_feat, snapshot_df)
    live_snapshot = base_snapshot.copy()
    for sym, vals in overrides.items():
        for feat, val in vals.items():
            live_snapshot[f"{sym}__{feat}"] = val
else:
    snapshot_df = pd.DataFrame()
    live_snapshot = base_snapshot.copy()
    snap_path = None

intraday = compute_intraday_context(live_snapshot, prev_snapshot, use_proxy)
live_snapshot["$NYMO__close_effective"] = intraday["nymo"]
live_snapshot["$NYSI__close_effective"] = intraday["nysi"]

# -----------------------------
# Score snapshot against learned gates
# -----------------------------
state_scores = {}
for state in ["bounce", "repair", "regime", "fall"]:
    state_scores[state] = score_state(live_snapshot, model["states"][state])

weekly_score = None
if weekly_feat is not None and not weekly_feat.empty and model.get("weekly_regime", {}).get("singles"):
    wk_base = latest_snapshot_dict(weekly_feat)
    weekly_score = score_state(wk_base, model["weekly_regime"])

canary = compute_canary_filter(live_snapshot, prev_snapshot)
decision = classify_signal(state_scores, intraday, canary, weekly_score)

# save upload history if snapshot uploaded
if snap_file is not None and snap_path is not None:
    append_upload_history({
        "upload_ts": datetime.now().isoformat(timespec="seconds"),
        "snapshot_file": str(snap_path),
        "verdict": decision["signal"],
        "bounce_prob": state_scores["bounce"]["prob"],
        "repair_prob": state_scores["repair"]["prob"],
        "regime_prob": state_scores["regime"]["prob"],
        "fall_prob": state_scores["fall"]["prob"],
    })

# -----------------------------
# Charts
# -----------------------------
def make_breadth_chart(hist_feat: pd.DataFrame, live_snapshot: Dict[str, float], sym: str) -> go.Figure:
    g = hist_feat[hist_feat["symbol"] == sym].sort_values("date").tail(180).copy()
    if g.empty:
        return go.Figure()
    live_close = live_snapshot.get(f"{sym}__close", np.nan)
    if pd.notna(live_close):
        g.loc[g.index[-1], "close"] = live_close
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g["date"], y=g["close"], name="Close", line=dict(width=2)))
    if "pct_b20" in g.columns:
        fig.add_trace(go.Scatter(x=g["date"], y=g["pct_b20"], name="%B", yaxis="y2", line=dict(width=1.5)))
    fig.update_layout(
        template="plotly_dark", height=350, margin=dict(l=20, r=20, t=30, b=20),
        yaxis=dict(title=sym), yaxis2=dict(title="%B", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h")
    )
    return fig

# -----------------------------
# Dashboard
# -----------------------------
tab1, tab2, tab3 = st.tabs(["Decision Dashboard", "Learned Gates", "History / Uploads"])

# Tab 1
with tab1:
    c1, c2, c3, c4 = st.columns([1.1, 1.1, 1.1, 1.3])
    with c1:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Daily Verdict</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-big">{decision["signal"]}</div>', unsafe_allow_html=True)
        st.markdown(color_pill(canary["label"]), unsafe_allow_html=True)
        st.markdown(color_pill(intraday["mode"]), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Bounce / Repair</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-big">{round(100*state_scores["bounce"]["prob"]):.0f} / {round(100*state_scores["repair"]["prob"]):.0f}</div>', unsafe_allow_html=True)
        st.caption("Historical probabilities")
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Regime / Fall</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-big">{round(100*state_scores["regime"]["prob"]):.0f} / {round(100*state_scores["fall"]["prob"]):.0f}</div>', unsafe_allow_html=True)
        st.caption("Historical probabilities")
        st.markdown("</div>", unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Intraday Proxy Context</div>', unsafe_allow_html=True)
        st.markdown(f"NYMO: **{fmt_num(intraday['nymo'])}**  \nΔ: **{fmt_num(intraday['nymo_delta'])}**")
        st.markdown(f"NYSI: **{fmt_num(intraday['nysi'])}**  \nΔ: **{fmt_num(intraday['nysi_delta'])}**")
        if weekly_score:
            st.caption(f"Weekly regime pass fraction: {weekly_score['pass_frac']:.0%}")
        st.markdown("</div>", unsafe_allow_html=True)

    action_cls = {"LONG": "action-long", "SHORT":"action-short", "HOLD":"action-hold"}[decision["signal"]]
    st.markdown(f'<div class="action-box {action_cls}"><b>{decision["signal"]}</b> — ' + " | ".join(decision["reasons"]) + "</div>", unsafe_allow_html=True)

    st.subheader("Gate Verdicts")
    verdict_cols = st.columns(4)
    for col, state in zip(verdict_cols, ["bounce", "repair", "regime", "fall"]):
        obj = state_scores[state]
        with col:
            st.metric(
                label=state.capitalize(),
                value=f"{round(100*obj['prob']):.0f}%",
                delta=f"Pass {obj['pass_frac']:.0%} | Base {obj['base_rate']:.0%}"
            )

    st.subheader("Why")
    for r in decision["reasons"]:
        st.write(f"• {r}")

    st.subheader("Gate Inputs")
    if decision["gate_params"]:
        gate_df = pd.DataFrame([{"Feature": k, "Current vs Gate": v} for k, v in decision["gate_params"].items()])
        st.dataframe(gate_df, use_container_width=True, hide_index=True)
    else:
        st.write("No strong gate alignment today.")

    st.subheader("Current Snapshot Core Readings")
    core = {
        "BPSPX %B": live_snapshot.get("$BPSPX__pct_b20"),
        "BPSPX": live_snapshot.get("$BPSPX__close"),
        "BPNYA": live_snapshot.get("$BPNYA__close"),
        "SPXA50R": live_snapshot.get("$SPXA50R__close"),
        "NYMO eff": intraday["nymo"],
        "NYSI eff": intraday["nysi"],
        "NYAD": live_snapshot.get("$NYAD__close"),
        "SPXADP": live_snapshot.get("$SPXADP__close"),
        "NYHL": live_snapshot.get("$NYHL__close"),
        "CPCE": live_snapshot.get("$CPCE__close"),
        "TRIN": live_snapshot.get("$TRIN__close"),
        "VIX": live_snapshot.get("VIX__close"),
        "VXX": live_snapshot.get("VXX__close"),
        "RSP:SPY": live_snapshot.get("RSP_SPY__close"),
    }
    core_df = pd.DataFrame({"Metric": list(core.keys()), "Value": [fmt_num(v, 3) for v in core.values()]})
    st.dataframe(core_df, use_container_width=True, hide_index=True)

    chart_symbol = st.selectbox("Breadth chart", options=["$BPSPX", "$SPXA50R", "$NYMO", "$NYSI", "$NYHL", "$TRIN", "RSP", "VIX", "VXX"], index=0)
    st.plotly_chart(make_breadth_chart(daily_feat, live_snapshot, chart_symbol), use_container_width=True)

# Tab 2
with tab2:
    st.subheader("Learned Hard Gates")
    choice = st.selectbox("State", options=["bounce", "repair", "regime", "fall", "weekly_regime"])
    if choice == "weekly_regime":
        bucket = model["weekly_regime"]
    else:
        bucket = model["states"][choice]
    st.caption(f"Base rate: {fmt_num(100*bucket['base_rate'],1)}%")
    sing = pd.DataFrame(bucket.get("singles", []))
    if not sing.empty:
        sing["gate"] = sing.apply(lambda r: gate_to_text(r.to_dict()), axis=1)
        sing = sing[["gate", "support", "hit_rate", "base_rate", "lift"]]
        sing["hit_rate"] = (100*sing["hit_rate"]).round(1)
        sing["base_rate"] = (100*sing["base_rate"]).round(1)
        sing["lift"] = sing["lift"].round(2)
        st.markdown("**Top single gates**")
        st.dataframe(sing, use_container_width=True, hide_index=True)
    combos = pd.DataFrame(bucket.get("combos", []))
    if not combos.empty:
        combos["gate"] = combos["gates"].apply(lambda gates: " AND ".join(gate_to_text(g) for g in gates))
        combos = combos[["gate", "support", "hit_rate", "base_rate", "lift"]]
        combos["hit_rate"] = (100*combos["hit_rate"]).round(1)
        combos["base_rate"] = (100*combos["base_rate"]).round(1)
        combos["lift"] = combos["lift"].round(2)
        st.markdown("**Top combo gates**")
        st.dataframe(combos, use_container_width=True, hide_index=True)

    st.subheader("Current state pass / fail")
    current_rows = []
    for state in ["bounce", "repair", "regime", "fall"]:
        obj = state_scores[state]
        for r in obj["single_results"]:
            current_rows.append({
                "State": state.capitalize(), "Gate": r["text"], "Passed": r["passed"], "Current": fmt_num(r["current"], 3), "HitRate%": round(100*r["hit_rate"], 1)
            })
    cur_df = pd.DataFrame(current_rows)
    st.dataframe(cur_df, use_container_width=True, hide_index=True)

# Tab 3
with tab3:
    st.subheader("Historical model summary")
    summary_df = pd.DataFrame({
        "Dataset": ["Daily baseline", "Weekly baseline"],
        "Rows": [len(daily_feat), len(weekly_feat) if weekly_feat is not None else 0],
        "Symbols": [daily_feat["symbol"].nunique(), weekly_feat["symbol"].nunique() if weekly_feat is not None and not weekly_feat.empty else 0],
        "LatestDate": [str(pd.to_datetime(daily_feat["date"]).max().date()), str(pd.to_datetime(weekly_feat["date"]).max().date()) if weekly_feat is not None and not weekly_feat.empty else "n/a"],
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.markdown("**Detected symbols**")
    st.write(", ".join(sorted(daily_feat["symbol"].unique().tolist())))

    hist = load_upload_history()
    if not hist.empty:
        st.subheader("Recent snapshot verdicts")
        st.dataframe(hist.tail(10), use_container_width=True, hide_index=True)
    else:
        st.write("No snapshot upload history yet.")
