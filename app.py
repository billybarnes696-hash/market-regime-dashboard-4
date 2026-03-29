
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# =============================
# Page setup / styling
# =============================
st.set_page_config(page_title="RSP Breadth Sweet Spot Engine", layout="wide", page_icon="📈")

CSS = """
<style>
:root{
  --bg:#08111f;
  --panel:#0e1a31;
  --panel2:#132445;
  --text:#ecf2ff;
  --muted:#94a7d3;
  --green:#22c55e;
  --yellow:#f59e0b;
  --red:#ef4444;
  --blue:#38bdf8;
  --purple:#a78bfa;
}
.block-container{padding-top:1rem;padding-bottom:2rem;}
.main-banner{
  background:linear-gradient(135deg, rgba(56,189,248,.18), rgba(167,139,250,.18));
  border:1px solid rgba(148,163,184,.25);
  border-radius:22px;
  padding:1rem 1.2rem;
  margin-bottom:1rem;
  box-shadow:0 16px 40px rgba(0,0,0,.18);
}
.soft-card{
  background:linear-gradient(180deg, rgba(14,26,49,.98), rgba(8,17,31,.99));
  border:1px solid rgba(148,163,184,.20);
  border-radius:18px;
  padding:1rem 1rem .95rem 1rem;
  box-shadow:0 10px 34px rgba(0,0,0,.20);
}
.score-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.9rem;}
.score-card{min-height:180px;display:flex;flex-direction:column;justify-content:space-between;}
.score-title{color:#c3d6ff;font-size:1.03rem;font-weight:800;letter-spacing:.02em;}
.score-value{font-size:3.15rem;line-height:1;font-weight:950;color:#fff;margin:.35rem 0;}
.score-sub{font-size:1rem;font-weight:800;}
.score-bar{width:100%;height:13px;border-radius:999px;background:rgba(255,255,255,.08);overflow:hidden;border:1px solid rgba(255,255,255,.08);margin-top:.55rem;}
.fill{height:100%;border-radius:999px;}
.fill-green{background:linear-gradient(90deg,#22c55e,#4ade80);}
.fill-yellow{background:linear-gradient(90deg,#f59e0b,#fbbf24);}
.fill-red{background:linear-gradient(90deg,#ef4444,#f87171);}
.fill-blue{background:linear-gradient(90deg,#38bdf8,#60a5fa);}
.kpi-row{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.75rem;margin-top:.8rem;}
.kpi-box{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);border-radius:14px;padding:.78rem .86rem;}
.kpi-label{color:#9cb1df;font-size:.88rem;font-weight:700;}
.kpi-value{color:#fff;font-size:1.85rem;font-weight:900;line-height:1.05;margin-top:.15rem;}
.pill{display:inline-block;padding:.28rem .6rem;border-radius:999px;font-size:.82rem;font-weight:800;border:1px solid rgba(255,255,255,.12);margin-right:.35rem;margin-bottom:.35rem;}
.pill-blue{background:rgba(56,189,248,.14);color:#d3efff;}
.pill-green{background:rgba(34,197,94,.14);color:#ccf8dc;}
.pill-yellow{background:rgba(245,158,11,.14);color:#fde8b0;}
.pill-red{background:rgba(239,68,68,.14);color:#ffd0d0;}
.action-box{border-radius:16px;padding:.86rem 1rem;margin:.55rem 0;border:1px solid rgba(255,255,255,.10);}
.action-existing{background:rgba(56,189,248,.10);}
.action-new{background:rgba(245,158,11,.10);}
.action-add{background:rgba(34,197,94,.10);}
.section-title{font-size:1.02rem;font-weight:850;color:#dbe8ff;margin-bottom:.45rem;}
.small-muted{color:#93a4cc;font-size:.9rem;}
.big-code pre{font-size:1.0rem !important;line-height:1.48 !important;}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    """
    <div class="main-banner">
      <div style="font-size:1.55rem;font-weight:950;">📈 RSP Breadth Sweet Spot Engine</div>
      <div class="small-muted">Empirical bounce / repair / regime model built from your historical StockCharts breadth data, with realtime snapshot updates and a canary-filter backtest.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================
# Persistence
# =============================
APP_DIR = Path("breadth_sweetspot_store")
APP_DIR.mkdir(exist_ok=True)
DAILY_PATH = APP_DIR / "daily_hist.parquet"
WEEKLY_PATH = APP_DIR / "weekly_hist.parquet"
MODEL_PATH = APP_DIR / "sweetspot_model.json"
UPLOAD_HISTORY_PATH = APP_DIR / "upload_history.csv"
SNAPSHOT_DIR = APP_DIR / "snapshots"
SNAPSHOT_DIR.mkdir(exist_ok=True)


# =============================
# Constants / symbol mapping
# =============================
PRIMARY_FEATURES = [
    "$BPSPX_bb20",
    "$BPNYA_close",
    "$OEXA200R_close",
    "$SPXA50R_close",
    "$NYMO_close",
    "$NYSI_close",
    "$CPCE_close",
    "$NYHL_close",
]

WEEKLY_FEATURES = [
    "$BPSPX_bb20",
    "$SPXA50R_close",
    "$NYSI_close",
    "$NYHL_close",
    "RSP_close",
    "RSP:SPY_close",
]

CHART_SYMBOLS = ["$BPSPX", "$BPNYA", "$SPXA50R", "$OEXA200R", "$NYMO", "$NYSI", "$CPCE", "$NYHL", "RSP", "RSP:SPY"]
MOMENTUM_SYMBOLS = ["$BPSPX_%B", "$BPNYA", "$SPXA50R", "$OEXA200R", "$NYMO", "$NYSI", "$CPCE", "$NYHL", "$NYAD", "$SPXADP", "RSP:SPY"]
CANARY_WEIGHTS = {
    "SPXS:SVOL": 0.24,
    "HYG:IEF": 0.20,
    "SMH:SPY": 0.18,
    "XLF:SPY": 0.12,
    "RSP:SPY": 0.12,
    "IWM:SPY": 0.10,
    "VXX": 0.04,   # inverted later
}

OUTCOME_DEFS = {
    "Bounce": {"horizon": 10, "ret": 0.03, "dd": -0.03, "type": "max"},
    "Repair": {"horizon": 20, "ret": 0.04, "dd": -0.05, "type": "end"},
    "Regime": {"horizon": 60, "ret": 0.08, "dd": -0.08, "type": "end"},
}
WEEKLY_REGIME_DEF = {"horizon": 12, "ret": 0.08, "dd": -0.08, "type": "end"}


# =============================
# Utilities
# =============================
def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return np.nan


def fmt_num(x, digits=2) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{float(x):.{digits}f}"


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def save_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2))


def normalize_symbol_from_filename(name: str) -> Tuple[str, str]:
    stem = Path(name).stem
    timeframe = "weekly" if stem.endswith(" w") else "daily"
    stem = stem.replace(" w", "")
    if stem.startswith("_"):
        base = stem[1:]
        return f"${base.upper()}", timeframe
    if "_" in stem:
        parts = [p.upper() for p in stem.split("_")]
        return ":".join(parts), timeframe
    return stem.upper(), timeframe


def now_eastern() -> pd.Timestamp:
    return pd.Timestamp.now(tz="America/New_York")


def market_phase(ts: pd.Timestamp | None = None) -> str:
    ts = ts or now_eastern()
    t = ts.hour + ts.minute / 60
    if t < 11:
        return "Morning"
    if t < 14:
        return "Midday"
    if t < 16:
        return "Power Hour"
    return "Post Close"


def official_mode(ts: pd.Timestamp | None = None) -> bool:
    ts = ts or now_eastern()
    return ts.hour >= 18


def score_color(score: float) -> Tuple[str, str]:
    if score >= 70:
        return "#22c55e", "fill-green"
    if score >= 45:
        return "#f59e0b", "fill-yellow"
    return "#ef4444", "fill-red"


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
    tp = (high + low + close) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))


def percent_b(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return (series - lower) / (upper - lower).replace(0, np.nan)


def roc(series: pd.Series, period: int = 3) -> pd.Series:
    return 100 * (series / series.shift(period) - 1)


def tsi(series: pd.Series, long: int = 25, short: int = 13, signal: int = 7) -> Tuple[pd.Series, pd.Series]:
    m = series.diff()
    a = m.abs()
    m1 = ema(ema(m, long), short)
    a1 = ema(ema(a, long), short)
    t = 100 * (m1 / a1.replace(0, np.nan))
    sig = ema(t, signal)
    return t, sig


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


def slope_n(series: pd.Series, n: int = 3) -> pd.Series:
    return series - series.shift(n)


# =============================
# Parsing / feature building
# =============================
@st.cache_data(show_spinner=False)
def parse_stockcharts_zip(file_bytes: bytes) -> Tuple[pd.DataFrame, pd.DataFrame]:
    daily_records = []
    weekly_records = []
    with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise ValueError("No CSV files were found inside the zip.")
        for name in names:
            symbol, timeframe = normalize_symbol_from_filename(name)
            raw = zf.read(name)
            df = pd.read_csv(io.BytesIO(raw), skiprows=1)
            df.columns = [str(c).strip() for c in df.columns]
            rename = {c: c.strip().title() for c in df.columns}
            df = df.rename(columns=rename)
            needed = {"Date", "Open", "High", "Low", "Close"}
            if not needed.issubset(set(df.columns)):
                continue
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            for c in ["Open", "High", "Low", "Close", "Volume"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["Date", "Close"]).copy()
            df["symbol"] = symbol
            df["timeframe"] = timeframe
            keep = ["Date", "symbol", "Open", "High", "Low", "Close", "Volume", "timeframe"]
            df = df[[c for c in keep if c in df.columns]].rename(columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
            if "volume" not in df.columns:
                df["volume"] = np.nan
            if timeframe == "weekly":
                weekly_records.append(df)
            else:
                daily_records.append(df)
    if not daily_records:
        raise ValueError("Could not parse daily history from the zip.")
    daily = pd.concat(daily_records, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    weekly = pd.concat(weekly_records, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True) if weekly_records else pd.DataFrame(columns=daily.columns)
    return daily, weekly


@st.cache_data(show_spinner=False)
def add_features(hist: pd.DataFrame) -> pd.DataFrame:
    if hist.empty:
        return hist.copy()
    frames = []
    for sym, g in hist.groupby("symbol", sort=False):
        g = g.sort_values("date").copy()
        g["rsi14"] = rsi(g["close"], 14)
        g["cci20"] = cci(g["high"], g["low"], g["close"], 20)
        g["bb20"] = percent_b(g["close"], 20, 2.0)
        g["roc3"] = roc(g["close"], 3)
        g["slope3"] = slope_n(g["close"], 3)
        g["tsi4"], g["tsi4_sig"] = tsi(g["close"], 4, 2, 4)
        g["ma20"] = g["close"].rolling(20).mean()
        g["ma50"] = g["close"].rolling(50).mean()
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def pivot_feature_frame(hist_feat: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    idx = None
    frame = pd.DataFrame()
    for sym, g in hist_feat.groupby("symbol", sort=False):
        s = g.sort_values("date").set_index("date")
        if idx is None:
            idx = s.index
        else:
            idx = idx.union(s.index)
    if idx is None:
        return pd.DataFrame()
    frame = pd.DataFrame(index=sorted(idx))
    for sym, g in hist_feat.groupby("symbol", sort=False):
        s = g.sort_values("date").set_index("date")
        if f"{sym}_close" in features:
            frame[f"{sym}_close"] = s["close"].reindex(frame.index)
        if f"{sym}_bb20" in features:
            frame[f"{sym}_bb20"] = s["bb20"].reindex(frame.index)
        if f"{sym}_rsi14" in features:
            frame[f"{sym}_rsi14"] = s["rsi14"].reindex(frame.index)
        if f"{sym}_roc3" in features:
            frame[f"{sym}_roc3"] = s["roc3"].reindex(frame.index)
    return frame.sort_index()


def compute_future_metrics(price: pd.Series, horizon: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    fwd_end = price.shift(-horizon) / price - 1
    max_gain = pd.Series(index=price.index, dtype=float)
    max_dd = pd.Series(index=price.index, dtype=float)
    vals = price.to_numpy(dtype=float)
    for i in range(len(vals)):
        if i + horizon >= len(vals):
            max_gain.iloc[i] = np.nan
            max_dd.iloc[i] = np.nan
            continue
        window = vals[i + 1 : i + horizon + 1]
        if len(window) == 0 or np.isnan(vals[i]):
            max_gain.iloc[i] = np.nan
            max_dd.iloc[i] = np.nan
            continue
        max_gain.iloc[i] = np.nanmax(window / vals[i] - 1)
        max_dd.iloc[i] = np.nanmin(window / vals[i] - 1)
    return fwd_end, max_gain, max_dd


def build_outcome_flags(rsp_price: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=rsp_price.index)
    for name, cfg in OUTCOME_DEFS.items():
        fwd_end, max_gain, max_dd = compute_future_metrics(rsp_price, cfg["horizon"])
        if cfg["type"] == "max":
            success = (max_gain >= cfg["ret"]) & (max_dd >= cfg["dd"])
        else:
            success = (fwd_end >= cfg["ret"]) & (max_dd >= cfg["dd"])
        out[f"{name}_success"] = success
        out[f"{name}_fwd_end"] = fwd_end
        out[f"{name}_max_gain"] = max_gain
        out[f"{name}_max_dd"] = max_dd
    return out


def build_weekly_regime_flags(rsp_weekly: pd.Series) -> pd.DataFrame:
    fwd_end, _, max_dd = compute_future_metrics(rsp_weekly, WEEKLY_REGIME_DEF["horizon"])
    success = (fwd_end >= WEEKLY_REGIME_DEF["ret"]) & (max_dd >= WEEKLY_REGIME_DEF["dd"])
    return pd.DataFrame({"WeeklyRegime_success": success, "WeeklyRegime_fwd_end": fwd_end, "WeeklyRegime_max_dd": max_dd}, index=rsp_weekly.index)


def band_distance_score(x: float, q25: float, med: float, q75: float) -> float:
    if pd.isna(x) or pd.isna(q25) or pd.isna(med) or pd.isna(q75):
        return np.nan
    iqr = max(abs(q75 - q25), 1e-6)
    if q25 <= x <= q75:
        d = abs(x - med) / iqr
        return max(0.72, 1.0 - 0.28 * d)
    d = min(abs(x - med) / iqr, 3.0)
    return max(0.0, 0.72 - 0.24 * (d - 1.0))


def build_combo_table(df: pd.DataFrame, outcome_col: str, summaries: pd.DataFrame, max_features: int = 4, min_support: int = 150) -> pd.DataFrame:
    available = [r["Feature"] for _, r in summaries.iterrows() if r["Feature"] in df.columns]
    rows = []
    base_hit = float(df[outcome_col].mean()) if len(df) else np.nan
    for k in [2, 3, 4]:
        if len(available) < k:
            continue
        from itertools import combinations
        for combo in combinations(available, k):
            mask = pd.Series(True, index=df.index)
            pieces = []
            for feat in combo:
                row = summaries.loc[summaries["Feature"] == feat].iloc[0]
                q25, q75 = row["Q25"], row["Q75"]
                mask &= df[feat].between(q25, q75, inclusive="both")
                pieces.append(f"{feat} [{q25:.3f},{q75:.3f}]")
            support = int(mask.sum())
            if support < min_support:
                continue
            hit = float(df.loc[mask, outcome_col].mean())
            if np.isnan(hit) or np.isnan(base_hit) or base_hit <= 0:
                continue
            rows.append({
                "Outcome": outcome_col.replace("_success", ""),
                "Combo": " AND ".join(pieces),
                "Support": support,
                "Hit Rate": hit,
                "Base Hit Rate": base_hit,
                "Lift": hit / base_hit,
                "Features": list(combo),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["Lift", "Hit Rate", "Support"], ascending=[False, False, False]).head(8).reset_index(drop=True)


def build_sweetspot_model(daily_feat: pd.DataFrame, weekly_feat: pd.DataFrame) -> Dict:
    daily_pivot = pivot_feature_frame(daily_feat, list(set(PRIMARY_FEATURES + ["RSP_close", "RSP:SPY_close"])))
    if "RSP_close" not in daily_pivot.columns:
        raise ValueError("RSP daily history is required for sweet-spot modeling.")
    rsp = daily_pivot["RSP_close"].dropna()
    outcomes = build_outcome_flags(rsp)
    base = daily_pivot.join(outcomes, how="inner").dropna(subset=["RSP_close"])

    summary_rows = []
    combo_frames = []
    for outcome in ["Bounce", "Repair", "Regime"]:
        outcome_col = f"{outcome}_success"
        success = base[base[outcome_col].fillna(False)].copy()
        for feat in PRIMARY_FEATURES:
            if feat not in base.columns:
                continue
            s = success[feat].dropna()
            u = base[feat].dropna()
            if len(s) < 50 or len(u) < 200:
                continue
            summary_rows.append({
                "Outcome": outcome,
                "Feature": feat,
                "Median": float(s.median()),
                "Q25": float(s.quantile(0.25)),
                "Q75": float(s.quantile(0.75)),
                "Universe Median": float(u.median()),
                "Hit Rate in Universe": float(base[outcome_col].mean()),
                "Universe Count": int(len(u)),
                "Success Count": int(len(s)),
            })
        sub = pd.DataFrame([r for r in summary_rows if r["Outcome"] == outcome])
        if not sub.empty:
            combo_frames.append(build_combo_table(base, outcome_col, sub))

    weekly_model = {}
    if not weekly_feat.empty:
        weekly_pivot = pivot_feature_frame(weekly_feat, list(set(WEEKLY_FEATURES + ["RSP_close"])))
        if "RSP_close" in weekly_pivot.columns:
            flags = build_weekly_regime_flags(weekly_pivot["RSP_close"].dropna())
            wb = weekly_pivot.join(flags, how="inner")
            rows = []
            success = wb[wb["WeeklyRegime_success"].fillna(False)]
            for feat in WEEKLY_FEATURES:
                if feat not in wb.columns:
                    continue
                s = success[feat].dropna()
                u = wb[feat].dropna()
                if len(s) < 20 or len(u) < 60:
                    continue
                rows.append({
                    "Outcome": "WeeklyRegime",
                    "Feature": feat,
                    "Median": float(s.median()),
                    "Q25": float(s.quantile(0.25)),
                    "Q75": float(s.quantile(0.75)),
                    "Universe Median": float(u.median()),
                    "Hit Rate in Universe": float(wb["WeeklyRegime_success"].mean()),
                    "Universe Count": int(len(u)),
                    "Success Count": int(len(s)),
                })
            weekly_model = {"summary": rows}

    summary_df = pd.DataFrame(summary_rows)
    combos_df = pd.concat(combo_frames, ignore_index=True) if combo_frames else pd.DataFrame(columns=["Outcome", "Combo", "Support", "Hit Rate", "Base Hit Rate", "Lift", "Features"])

    return {
        "summary": summary_df.to_dict(orient="records"),
        "combos": combos_df.to_dict(orient="records"),
        "weekly": weekly_model,
        "features": PRIMARY_FEATURES,
        "built_at": datetime.utcnow().isoformat(),
    }


def save_snapshot_file(df: pd.DataFrame) -> str:
    path = SNAPSHOT_DIR / f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(path, index=False)
    return str(path)


def load_upload_history() -> pd.DataFrame:
    if not UPLOAD_HISTORY_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(UPLOAD_HISTORY_PATH)
    except Exception:
        return pd.DataFrame()


def append_upload_history(row: Dict):
    hist = load_upload_history()
    hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    hist.to_csv(UPLOAD_HISTORY_PATH, index=False)


# =============================
# Realtime snapshot / current state
# =============================
@st.cache_data(show_spinner=False)
def parse_realtime_snapshot(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    cols = {c.lower().strip(): c for c in df.columns}
    if "symbol" not in cols:
        raise ValueError("Realtime snapshot must include a Symbol column.")
    close_col = None
    for cand in ["close", "last", "price", "current", "value", "daily close", "close price"]:
        if cand in cols:
            close_col = cols[cand]
            break
    if close_col is None:
        raise ValueError("Realtime snapshot must include a close-like column.")
    out = pd.DataFrame({
        "Symbol": df[cols["symbol"]].astype(str).str.strip(),
        "Close": pd.to_numeric(df[close_col], errors="coerce"),
    })
    pct_col = None
    for cand in ["daily pctchange(1,daily close)", "% change", "pct change", "change %", "daily change %"]:
        if cand in cols:
            pct_col = cols[cand]
            break
    out["PctChange"] = pd.to_numeric(df[pct_col], errors="coerce") if pct_col else np.nan
    return out.dropna(subset=["Close"])


def latest_feature_snapshot(hist_feat: pd.DataFrame, prior: bool = False) -> Dict[str, float]:
    snap = {}
    if hist_feat.empty:
        return snap
    for sym, g in hist_feat.groupby("symbol", sort=False):
        g = g.sort_values("date")
        if prior and len(g) < 2:
            continue
        row = g.iloc[-2] if prior else g.iloc[-1]
        snap[sym] = safe_float(row.get("close"))
        snap[f"{sym}_%B"] = safe_float(row.get("bb20"))
        snap[f"{sym}_RSI14"] = safe_float(row.get("rsi14"))
        snap[f"{sym}_CCI20"] = safe_float(row.get("cci20"))
        snap[f"{sym}_ROC3"] = safe_float(row.get("roc3"))
        snap[f"{sym}_SLOPE3"] = safe_float(row.get("slope3"))
    return snap


def recompute_with_realtime(hist_feat: pd.DataFrame, realtime_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    out = {}
    for _, row in realtime_df.iterrows():
        sym = str(row["Symbol"]).strip()
        new_close = safe_float(row["Close"])
        g = hist_feat[hist_feat["symbol"] == sym].sort_values("date").copy()
        if g.empty or pd.isna(new_close):
            continue
        g.iloc[-1, g.columns.get_loc("close")] = new_close
        g.iloc[-1, g.columns.get_loc("high")] = max(safe_float(g.iloc[-1]["high"]), new_close)
        g.iloc[-1, g.columns.get_loc("low")] = min(safe_float(g.iloc[-1]["low"]), new_close)
        if len(g) > 1:
            g.iloc[-1, g.columns.get_loc("open")] = safe_float(g.iloc[-2]["close"])
        g = add_features(g)
        last = g.iloc[-1]
        out[sym] = {
            "close": safe_float(last["close"]),
            "%B": safe_float(last["bb20"]),
            "RSI14": safe_float(last["rsi14"]),
            "CCI20": safe_float(last["cci20"]),
            "ROC3": safe_float(last["roc3"]),
            "SLOPE3": safe_float(last["slope3"]),
        }
    return out


def proxy_nymo(snapshot: Dict[str, float], prev_snapshot: Dict[str, float]) -> Dict[str, float]:
    nyad = safe_float(snapshot.get("$NYAD"))
    spxadp = safe_float(snapshot.get("$SPXADP"))
    prev_nyad = safe_float(prev_snapshot.get("$NYAD"))
    prev_spxadp = safe_float(prev_snapshot.get("$SPXADP"))
    if pd.isna(nyad) and pd.isna(spxadp):
        return {"value": np.nan, "delta": np.nan, "state": "Unavailable"}
    cur_raw = 0.6 * (0 if pd.isna(nyad) else nyad) + 0.4 * (0 if pd.isna(spxadp) else spxadp)
    prev_raw = 0.6 * (0 if pd.isna(prev_nyad) else prev_nyad) + 0.4 * (0 if pd.isna(prev_spxadp) else prev_spxadp)
    cur = 100 * np.tanh(cur_raw / 1600.0)
    prev = 100 * np.tanh(prev_raw / 1600.0)
    delta = cur - prev
    if cur <= -70:
        state = "Deep washout"
    elif cur <= -20:
        state = "Negative but repairing" if delta > 0 else "Negative and weak"
    elif cur <= 20:
        state = "Neutral / crossing"
    else:
        state = "Positive thrust"
    return {"value": cur, "delta": delta, "state": state}


@dataclass
class CurrentState:
    snapshot: Dict[str, float]
    prev_snapshot: Dict[str, float]
    source_mode: str
    market_phase: str


def build_current_state(daily_feat: pd.DataFrame, weekly_feat: pd.DataFrame, realtime_df: pd.DataFrame | None) -> CurrentState:
    snap = latest_feature_snapshot(daily_feat, prior=False)
    prev = latest_feature_snapshot(daily_feat, prior=True)
    if realtime_df is not None and not realtime_df.empty:
        updates = recompute_with_realtime(daily_feat, realtime_df)
        for sym, vals in updates.items():
            snap[sym] = vals["close"]
            snap[f"{sym}_%B"] = vals["%B"]
            snap[f"{sym}_RSI14"] = vals["RSI14"]
            snap[f"{sym}_CCI20"] = vals["CCI20"]
            snap[f"{sym}_ROC3"] = vals["ROC3"]
            snap[f"{sym}_SLOPE3"] = vals["SLOPE3"]
    # weekly latest snapshot into same namespace
    if not weekly_feat.empty:
        for sym, g in weekly_feat.groupby("symbol", sort=False):
            g = g.sort_values("date")
            row = g.iloc[-1]
            snap[f"W_{sym}"] = safe_float(row.get("close"))
            snap[f"W_{sym}_%B"] = safe_float(row.get("bb20"))
    ts = now_eastern()
    phase = market_phase(ts)
    official = official_mode(ts)
    pnymo = proxy_nymo(snap, prev)
    if official and pd.notna(snap.get("$NYMO", np.nan)):
        snap["$NYMO_EFFECTIVE"] = safe_float(snap.get("$NYMO"))
        snap["$NYMO_EFFECTIVE_DELTA"] = safe_float(snap.get("$NYMO")) - safe_float(prev.get("$NYMO"))
        mode = "Official breadth mode"
    else:
        snap["$NYMO_EFFECTIVE"] = pnymo["value"]
        snap["$NYMO_EFFECTIVE_DELTA"] = pnymo["delta"]
        mode = "Intraday proxy mode"
    snap["$NYMO_PROXY_STATE"] = pnymo["state"]
    return CurrentState(snapshot=snap, prev_snapshot=prev, source_mode=mode, market_phase=phase)


# =============================
# Scoring
# =============================
def model_tables(model: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = pd.DataFrame(model.get("summary", []))
    combos = pd.DataFrame(model.get("combos", []))
    weekly = pd.DataFrame(model.get("weekly", {}).get("summary", []))
    return summary, combos, weekly


def current_feature_value(feature: str, snapshot: Dict[str, float]) -> float:
    if feature.endswith("_close"):
        sym = feature.replace("_close", "")
        if sym == "$NYMO":
            return safe_float(snapshot.get("$NYMO_EFFECTIVE"))
        return safe_float(snapshot.get(sym))
    if feature.endswith("_bb20"):
        sym = feature.replace("_bb20", "")
        return safe_float(snapshot.get(f"{sym}_%B"))
    return np.nan


def evaluate_outcome_score(outcome: str, summary_df: pd.DataFrame, combo_df: pd.DataFrame, snapshot: Dict[str, float]) -> Tuple[float, pd.DataFrame, Dict]:
    sub = summary_df[summary_df["Outcome"] == outcome].copy()
    rows = []
    scores = []
    for _, r in sub.iterrows():
        cur = current_feature_value(r["Feature"], snapshot)
        sc = band_distance_score(cur, r["Q25"], r["Median"], r["Q75"])
        scores.append(sc)
        rows.append({
            "Feature": r["Feature"],
            "Current": cur,
            "Median": r["Median"],
            "Q25": r["Q25"],
            "Q75": r["Q75"],
            "Fit": sc,
        })
    detail = pd.DataFrame(rows)
    core = float(np.nanmean(scores) * 100) if len(scores) else np.nan
    combo_bonus = 0.0
    combo_match = "None"
    combo_sub = combo_df[combo_df["Outcome"] == outcome].copy() if not combo_df.empty else pd.DataFrame()
    if not combo_sub.empty:
        for _, r in combo_sub.iterrows():
            ok = True
            for feat in r.get("Features", []):
                rr = sub[sub["Feature"] == feat]
                if rr.empty:
                    ok = False
                    break
                rr = rr.iloc[0]
                cur = current_feature_value(feat, snapshot)
                if pd.isna(cur) or not (rr["Q25"] <= cur <= rr["Q75"]):
                    ok = False
                    break
            if ok:
                combo_bonus = min(12.0, max(combo_bonus, 12.0 * min(1.0, float(r["Lift"]) - 1.0 + 0.3)))
                combo_match = r["Combo"]
                break
    total = float(min(100.0, (0 if np.isnan(core) else core) + combo_bonus))
    meta = {"combo_bonus": combo_bonus, "combo_match": combo_match}
    return total, detail, meta


def evaluate_weekly_overlay(weekly_df: pd.DataFrame, snapshot: Dict[str, float]) -> Tuple[float, pd.DataFrame]:
    rows = []
    scores = []
    for _, r in weekly_df.iterrows():
        feat = r["Feature"]
        if feat.endswith("_close"):
            sym = feat.replace("_close", "")
            cur = safe_float(snapshot.get(f"W_{sym}", snapshot.get(sym)))
        elif feat.endswith("_bb20"):
            sym = feat.replace("_bb20", "")
            cur = safe_float(snapshot.get(f"W_{sym}_%B"))
        else:
            cur = np.nan
        sc = band_distance_score(cur, r["Q25"], r["Median"], r["Q75"])
        scores.append(sc)
        rows.append({"Feature": feat, "Current": cur, "Median": r["Median"], "Q25": r["Q25"], "Q75": r["Q75"], "Fit": sc})
    detail = pd.DataFrame(rows)
    return float(np.nanmean(scores) * 100) if len(scores) else np.nan, detail


def action_hierarchy(bounce: float, repair: float, regime: float, weekly: float, snapshot: Dict[str, float], canary_comp: float, canary_conf: float) -> Dict[str, str | float]:
    existing = "Stay defensive / monitor"
    new = "No new long"
    add = "Do not add"
    rsp_size = 0.0
    ursp_size = 0.0

    bpspx_bb = safe_float(snapshot.get("$BPSPX_%B"))
    spxa50r = safe_float(snapshot.get("$SPXA50R"))
    nymo = safe_float(snapshot.get("$NYMO_EFFECTIVE"))
    if regime >= 72 and weekly >= 65 and canary_comp > 0.05 and canary_conf >= 55 and bpspx_bb >= 0.45 and spxa50r >= 55 and nymo > -10:
        existing = "Keep long bias"
        new = "New RSP okay; URSP allowed selectively"
        add = "Add on hold / breakout confirmation"
        rsp_size = 0.30
        ursp_size = 0.10
    elif repair >= 60 and bounce >= 45 and weekly >= 45 and canary_comp > -0.05:
        existing = "Hold / keep existing probe"
        new = "New RSP okay"
        add = "Add only if breadth keeps improving"
        rsp_size = 0.20
    elif bounce >= 50:
        existing = "Small probe only"
        new = "New probe RSP only"
        add = "No adds yet"
        rsp_size = 0.10
    return {
        "existing": existing,
        "new": new,
        "add": add,
        "rsp_size": rsp_size,
        "ursp_size": ursp_size,
    }


def classify_delta(sym: str, cur: float, prev: float) -> Tuple[str, str]:
    if pd.isna(cur) or pd.isna(prev):
        return "n/a", "missing"
    delta = cur - prev
    a = abs(delta)
    # symbol-specific thresholds
    if sym == "$BPSPX_%B":
        if a >= 0.20:
            return ("Shock+", "vertical repair") if delta > 0 else ("Shock-", "vertical damage")
        if a >= 0.08:
            return ("Thrust", "repairing quickly") if delta > 0 else ("Collapse", "breaking quickly")
    elif sym in {"$SPXA50R", "$OEXA200R", "$BPNYA", "$NYHL"}:
        if a >= 15:
            return ("Shock+", "vertical repair") if delta > 0 else ("Shock-", "vertical damage")
        if a >= 6:
            return ("Thrust", "repairing quickly") if delta > 0 else ("Collapse", "breaking quickly")
    elif sym in {"$NYMO", "$NYMO_EFFECTIVE"}:
        if a >= 30:
            return ("Shock+", "vertical repair") if delta > 0 else ("Shock-", "vertical damage")
        if a >= 12:
            return ("Thrust", "repairing quickly") if delta > 0 else ("Collapse", "breaking quickly")
    elif sym in {"$NYSI"}:
        if a >= 120:
            return ("Shock+", "vertical repair") if delta > 0 else ("Shock-", "vertical damage")
        if a >= 40:
            return ("Thrust", "repairing quickly") if delta > 0 else ("Collapse", "breaking quickly")
    elif sym in {"$CPCE"}:
        if a >= 0.15:
            return ("Shock+", "fear spike") if delta > 0 else ("Shock-", "fear unwind")
        if a >= 0.06:
            return ("Thrust", "fear building") if delta > 0 else ("Collapse", "fear fading")
    elif sym in {"$NYAD", "$SPXADP"}:
        if a >= 1500 if sym == "$NYAD" else a >= 60:
            return ("Shock+", "breadth thrust") if delta > 0 else ("Shock-", "breadth washout")
        if a >= 600 if sym == "$NYAD" else a >= 25:
            return ("Thrust", "broadening") if delta > 0 else ("Collapse", "narrowing")
    elif sym in {"RSP:SPY"}:
        if a >= 0.02:
            return ("Shock+", "leadership spike") if delta > 0 else ("Shock-", "leadership fade")
        if a >= 0.006:
            return ("Thrust", "leadership improving") if delta > 0 else ("Collapse", "leadership weakening")
    if delta > 0:
        return "Improve", "grinding higher"
    if delta < 0:
        return "Fade", "rolling over"
    return "Flat", "flat"


def momentum_table(snapshot: Dict[str, float], prev_snapshot: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for sym in MOMENTUM_SYMBOLS:
        if sym.endswith("_%B"):
            cur = safe_float(snapshot.get(sym))
            prev = safe_float(prev_snapshot.get(sym))
        else:
            actual = "$NYMO_EFFECTIVE" if sym == "$NYMO" else sym
            cur = safe_float(snapshot.get(actual))
            prev = safe_float(prev_snapshot.get(sym)) if sym != "$NYMO" else safe_float(prev_snapshot.get("$NYMO"))
        label, state = classify_delta(sym.replace("$NYMO", "$NYMO_EFFECTIVE"), cur, prev)
        rows.append({
            "Symbol": sym,
            "Current": None if pd.isna(cur) else round(float(cur), 3),
            "Prior": None if pd.isna(prev) else round(float(prev), 3),
            "Delta": None if pd.isna(cur) or pd.isna(prev) else round(float(cur - prev), 3),
            "Δ Class": label,
            "State": state,
        })
    return pd.DataFrame(rows)


# =============================
# Canary overlay from uploaded history
# =============================
def ratio_indicator_score(close: pd.Series) -> pd.Series:
    if close.dropna().shape[0] < 220:
        return pd.Series(dtype=float)
    mh = macd_hist(close)
    t, _ = tsi(close, 40, 20, 10)
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
    piv = pivot_feature_frame(daily_feat, [f"{s}_close" for s in CANARY_WEIGHTS.keys()])
    score_map = {}
    for sym, wt in CANARY_WEIGHTS.items():
        col = f"{sym}_close"
        if col not in piv.columns:
            continue
        s = piv[col].dropna()
        sc = ratio_indicator_score(s)
        if sc.empty:
            continue
        # VXX inverted: rising vol is bad
        if sym == "VXX":
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
    align = (np.sign(df).replace(0, np.nan).eq(sign, axis=0)).mean(axis=1).fillna(0)
    strength = df.abs().mean(axis=1).fillna(0)
    conf = ((0.6 * align) + (0.4 * strength)) * 100.0
    out = pd.DataFrame({"canary_comp": comp, "canary_conf": conf})
    return out


# =============================
# Historical score series / backtest
# =============================
def build_score_series(daily_feat: pd.DataFrame, weekly_feat: pd.DataFrame, model: Dict) -> pd.DataFrame:
    summary_df, combo_df, weekly_df = model_tables(model)
    daily_pivot = pivot_feature_frame(daily_feat, list(set(PRIMARY_FEATURES + ["RSP_close"]))).dropna(subset=["RSP_close"]).copy()
    if daily_pivot.empty:
        return pd.DataFrame()
    # add official NYMO effective placeholder and map required columns back into snapshot per date
    canary = build_canary_from_history(daily_feat)
    weekly_pivot = pivot_feature_frame(weekly_feat, list(set(WEEKLY_FEATURES + ["RSP_close"]))) if not weekly_feat.empty else pd.DataFrame()
    weekly_pivot = weekly_pivot.reindex(daily_pivot.index, method="ffill") if not weekly_pivot.empty else weekly_pivot

    rows = []
    prev_snapshot = None
    for dt, row in daily_pivot.iterrows():
        snapshot = {}
        for feat in PRIMARY_FEATURES:
            if feat in daily_pivot.columns:
                if feat.endswith("_close"):
                    snapshot[feat.replace("_close", "")] = row.get(feat)
                elif feat.endswith("_bb20"):
                    snapshot[f"{feat.replace('_bb20', '')}_%B"] = row.get(feat)
        snapshot["RSP"] = row.get("RSP_close")
        if "$NYMO" in snapshot:
            snapshot["$NYMO_EFFECTIVE"] = snapshot["$NYMO"]
        if not weekly_pivot.empty and dt in weekly_pivot.index:
            wrow = weekly_pivot.loc[dt]
            for feat in WEEKLY_FEATURES:
                if feat in weekly_pivot.columns:
                    if feat.endswith("_close"):
                        snapshot[f"W_{feat.replace('_close', '')}"] = wrow.get(feat)
                    elif feat.endswith("_bb20"):
                        snapshot[f"W_{feat.replace('_bb20', '')}_%B"] = wrow.get(feat)
        bounce, _, _ = evaluate_outcome_score("Bounce", summary_df, combo_df, snapshot)
        repair, _, _ = evaluate_outcome_score("Repair", summary_df, combo_df, snapshot)
        regime, _, _ = evaluate_outcome_score("Regime", summary_df, combo_df, snapshot)
        weekly_score, _ = evaluate_weekly_overlay(weekly_df, snapshot) if not weekly_df.empty else (np.nan, pd.DataFrame())
        canary_comp = canary.loc[dt, "canary_comp"] if not canary.empty and dt in canary.index else np.nan
        canary_conf = canary.loc[dt, "canary_conf"] if not canary.empty and dt in canary.index else np.nan
        master = 0.35 * bounce + 0.40 * repair + 0.25 * regime
        if pd.notna(weekly_score):
            master = 0.75 * master + 0.25 * weekly_score
        rows.append({
            "date": dt,
            "rsp_close": row.get("RSP_close"),
            "bounce_score": bounce,
            "repair_score": repair,
            "regime_score": regime,
            "weekly_score": weekly_score,
            "master_score": master,
            "canary_comp": canary_comp,
            "canary_conf": canary_conf,
        })
    return pd.DataFrame(rows)


def run_backtest(score_df: pd.DataFrame, fast: int = 5, slow: int = 13, deadband: float = 0.0, use_canary: bool = True, canary_thr: float = 0.05, conf_thr: float = 55.0, switch_cost_bps: float = 5.0) -> pd.DataFrame:
    bt = score_df.copy().sort_values("date").reset_index(drop=True)
    bt["ema_fast"] = ema(bt["master_score"], fast)
    bt["ema_slow"] = ema(bt["master_score"], slow)
    bt["osc"] = (bt["ema_fast"] - bt["ema_slow"]) / bt["ema_slow"].replace(0, np.nan)
    signal = bt["osc"] > deadband
    if use_canary:
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


# =============================
# Rendering helpers
# =============================
def render_score_card(title: str, score: float, subtitle: str, meta: str = ""):
    hex_color, fill = score_color(score)
    st.markdown(
        f"""
        <div class="soft-card score-card">
          <div>
            <div class="score-title">{title}</div>
            <div class="score-value">{score:.0f}</div>
            <div class="score-sub" style="color:{hex_color};">{subtitle}</div>
            <div class="small-muted" style="margin-top:.35rem;">{meta}</div>
          </div>
          <div class="score-bar"><div class="fill {fill}" style="width:{max(0,min(100,score))}%;"></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_action_box(label: str, value: str, cls: str):
    st.markdown(f'<div class="action-box {cls}"><div class="section-title">{label}</div><div>{value}</div></div>', unsafe_allow_html=True)


def plot_symbol_panel(hist_feat: pd.DataFrame, symbol: str, current_override: float | None = None):
    g = hist_feat[hist_feat["symbol"] == symbol].sort_values("date").tail(220).copy()
    if g.empty:
        return None
    if current_override is not None and not pd.isna(current_override):
        g.iloc[-1, g.columns.get_loc("close")] = current_override
        g = add_features(g)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=g["date"], open=g["open"], high=g["high"], low=g["low"], close=g["close"], name=symbol), row=1, col=1)
    fig.add_trace(go.Scatter(x=g["date"], y=g["ma20"], name="MA20", line=dict(width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=g["date"], y=g["ma50"], name="MA50", line=dict(width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=g["date"], y=g["bb20"], name="%B", line=dict(width=1.8)), row=2, col=1)
    fig.add_trace(go.Scatter(x=g["date"], y=g["rsi14"], name="RSI14", line=dict(width=1.3)), row=2, col=1)
    fig.update_layout(height=500, margin=dict(l=10, r=10, t=30, b=10), xaxis_rangeslider_visible=False, template="plotly_dark", legend_orientation="h")
    return fig


# =============================
# Sidebar / data loading
# =============================
with st.sidebar:
    st.markdown("### Data")
    historical_zip = st.file_uploader("One-time historical zip", type=["zip"], help="Upload the full StockCharts zip once. Daily files have no 'w'; weekly files do.")
    realtime_csv = st.file_uploader("Realtime snapshot CSV", type=["csv"], help="Upload your current realtime breadth snapshot whenever you want to refresh the dashboard.")
    rebuild = st.button("Rebuild historical model", use_container_width=True)
    reset = st.button("Reset local store", use_container_width=True)
    st.markdown("---")
    st.markdown("### Backtest settings")
    fast = st.slider("Oscillator fast EMA", 3, 15, 5)
    slow = st.slider("Oscillator slow EMA", 8, 34, 13)
    deadband = st.slider("Oscillator deadband", 0.0, 0.10, 0.0, 0.005)
    use_canary = st.toggle("Use canary filter", value=True)
    canary_thr = st.slider("Canary composite threshold", -0.20, 0.30, 0.05, 0.01)
    conf_thr = st.slider("Canary confidence threshold", 0, 100, 55)

if reset:
    for p in [DAILY_PATH, WEEKLY_PATH, MODEL_PATH, UPLOAD_HISTORY_PATH]:
        if p.exists():
            p.unlink()
    for p in SNAPSHOT_DIR.glob("*.csv"):
        p.unlink()
    st.success("Local store reset.")

if historical_zip is not None and (rebuild or (not DAILY_PATH.exists()) or (not MODEL_PATH.exists())):
    with st.spinner("Parsing historical zip and building sweet-spot model..."):
        daily_raw, weekly_raw = parse_stockcharts_zip(historical_zip.getvalue())
        daily_feat = add_features(daily_raw)
        weekly_feat = add_features(weekly_raw)
        model = build_sweetspot_model(daily_feat, weekly_feat)
        daily_feat.to_parquet(DAILY_PATH, index=False)
        weekly_feat.to_parquet(WEEKLY_PATH, index=False)
        save_json(MODEL_PATH, model)
    st.success("Historical sweet-spot model rebuilt.")

if not DAILY_PATH.exists() or not MODEL_PATH.exists():
    st.info("Upload your historical StockCharts zip in the sidebar to initialize the engine.")
    st.stop()

daily_feat = pd.read_parquet(DAILY_PATH)
weekly_feat = pd.read_parquet(WEEKLY_PATH) if WEEKLY_PATH.exists() else pd.DataFrame(columns=daily_feat.columns)
model = load_json(MODEL_PATH, {})
summary_df, combo_df, weekly_model_df = model_tables(model)

realtime_df = None
if realtime_csv is not None:
    realtime_df = parse_realtime_snapshot(realtime_csv.getvalue())
    snap_path = save_snapshot_file(realtime_df)
else:
    hist = load_upload_history()
    if not hist.empty and "snapshot_file" in hist.columns:
        p = Path(hist.sort_values("upload_ts").iloc[-1]["snapshot_file"])
        if p.exists():
            realtime_df = pd.read_csv(p)
            snap_path = str(p)
        else:
            snap_path = ""
    else:
        snap_path = ""

state = build_current_state(daily_feat, weekly_feat, realtime_df)

bounce_score, bounce_detail, bounce_meta = evaluate_outcome_score("Bounce", summary_df, combo_df, state.snapshot)
repair_score, repair_detail, repair_meta = evaluate_outcome_score("Repair", summary_df, combo_df, state.snapshot)
regime_score, regime_detail, regime_meta = evaluate_outcome_score("Regime", summary_df, combo_df, state.snapshot)
weekly_overlay, weekly_detail = evaluate_weekly_overlay(weekly_model_df, state.snapshot) if not weekly_model_df.empty else (np.nan, pd.DataFrame())

score_df = build_score_series(daily_feat, weekly_feat, model)
canary_hist = build_canary_from_history(daily_feat)
if not canary_hist.empty:
    last_canary_comp = float(canary_hist.iloc[-1]["canary_comp"])
    last_canary_conf = float(canary_hist.iloc[-1]["canary_conf"])
else:
    last_canary_comp = np.nan
    last_canary_conf = np.nan

actions = action_hierarchy(bounce_score, repair_score, regime_score, weekly_overlay, state.snapshot, last_canary_comp, last_canary_conf)
master_score = 0.35 * bounce_score + 0.40 * repair_score + 0.25 * regime_score
if pd.notna(weekly_overlay):
    master_score = 0.75 * master_score + 0.25 * weekly_overlay

if realtime_df is not None and not realtime_df.empty:
    append_upload_history({
        "upload_ts": datetime.now().isoformat(timespec="seconds"),
        "snapshot_file": snap_path,
        "master_score": master_score,
        "bounce_score": bounce_score,
        "repair_score": repair_score,
        "regime_score": regime_score,
        "weekly_overlay": weekly_overlay,
    })

# =============================
# Main tabs
# =============================
tab1, tab2, tab3 = st.tabs(["Decision Dashboard", "Backtest + Oscillator", "Sweet Spot Explorer"])

with tab1:
    left, right = st.columns([1.35, 0.95], gap="large")
    with left:
        st.markdown('<div class="score-grid">', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            render_score_card("Bounce Score", bounce_score, "Oversold-to-bounce fit", f"Combo bonus: {bounce_meta['combo_bonus']:.1f}")
        with c2:
            render_score_card("Repair Score", repair_score, "Breadth repair fit", f"Combo bonus: {repair_meta['combo_bonus']:.1f}")
        c3, c4 = st.columns(2)
        with c3:
            render_score_card("Regime Score", regime_score, "Durable participation fit", f"Combo bonus: {regime_meta['combo_bonus']:.1f}")
        with c4:
            render_score_card("Weekly Overlay", 0 if pd.isna(weekly_overlay) else weekly_overlay, "Weekly backdrop", "Uses weekly historical sweet spots")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">At a Glance</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="kpi-row">
              <div class="kpi-box"><div class="kpi-label">Master Score</div><div class="kpi-value">{master_score:.0f}</div></div>
              <div class="kpi-box"><div class="kpi-label">Market Phase</div><div class="kpi-value" style="font-size:1.35rem;">{state.market_phase}</div></div>
              <div class="kpi-box"><div class="kpi-label">Breadth Source</div><div class="kpi-value" style="font-size:1.1rem;">{state.source_mode}</div></div>
              <div class="kpi-box"><div class="kpi-label">Canary</div><div class="kpi-value">{fmt_num(last_canary_comp,2)} / {fmt_num(last_canary_conf,0)}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        pills = []
        pills.append(f'<span class="pill pill-blue">NYMO effective: {fmt_num(state.snapshot.get("$NYMO_EFFECTIVE"),1)}</span>')
        pills.append(f'<span class="pill pill-yellow">BPSPX %B: {fmt_num(state.snapshot.get("$BPSPX_%B"),2)}</span>')
        pills.append(f'<span class="pill pill-green">SPXA50R: {fmt_num(state.snapshot.get("$SPXA50R"),1)}</span>')
        pills.append(f'<span class="pill pill-red">CPCE: {fmt_num(state.snapshot.get("$CPCE"),2)}</span>')
        st.markdown("".join(pills), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### Current Action Hierarchy")
        render_action_box("Existing", actions["existing"], "action-existing")
        render_action_box("New", actions["new"], "action-new")
        render_action_box("Add", actions["add"], "action-add")
        st.markdown(
            f"<div class='small-muted'>Suggested sizing: RSP {actions['rsp_size']*100:.0f}% | URSP {actions['ursp_size']*100:.0f}%</div>",
            unsafe_allow_html=True,
        )

        st.markdown("### Momentum Context")
        st.dataframe(momentum_table(state.snapshot, state.prev_snapshot), use_container_width=True, hide_index=True)

    with right:
        st.markdown("### Sweet Spot Match Detail")
        subtab1, subtab2, subtab3, subtab4 = st.tabs(["Bounce", "Repair", "Regime", "Weekly"])
        with subtab1:
            show = bounce_detail.copy()
            for col in ["Current", "Median", "Q25", "Q75", "Fit"]:
                show[col] = pd.to_numeric(show[col], errors="coerce").round(3)
            st.dataframe(show, use_container_width=True, hide_index=True)
            st.caption(f"Best current combo match: {bounce_meta['combo_match']}")
        with subtab2:
            show = repair_detail.copy()
            for col in ["Current", "Median", "Q25", "Q75", "Fit"]:
                show[col] = pd.to_numeric(show[col], errors="coerce").round(3)
            st.dataframe(show, use_container_width=True, hide_index=True)
            st.caption(f"Best current combo match: {repair_meta['combo_match']}")
        with subtab3:
            show = regime_detail.copy()
            for col in ["Current", "Median", "Q25", "Q75", "Fit"]:
                show[col] = pd.to_numeric(show[col], errors="coerce").round(3)
            st.dataframe(show, use_container_width=True, hide_index=True)
            st.caption(f"Best current combo match: {regime_meta['combo_match']}")
        with subtab4:
            if weekly_detail.empty:
                st.info("No weekly model available from the uploaded history.")
            else:
                show = weekly_detail.copy()
                for col in ["Current", "Median", "Q25", "Q75", "Fit"]:
                    show[col] = pd.to_numeric(show[col], errors="coerce").round(3)
                st.dataframe(show, use_container_width=True, hide_index=True)

        st.markdown("### Chart Panel")
        chart_sym = st.selectbox("Symbol", CHART_SYMBOLS, index=0)
        override = state.snapshot.get(chart_sym)
        fig = plot_symbol_panel(daily_feat, chart_sym, override)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    if score_df.empty:
        st.warning("Not enough data to build the historical oscillator.")
    else:
        bt = run_backtest(score_df, fast=fast, slow=slow, deadband=deadband, use_canary=use_canary, canary_thr=canary_thr, conf_thr=conf_thr)
        total_ret = bt["equity_strategy"].iloc[-1] - 1
        bh_ret = bt["equity_buyhold"].iloc[-1] - 1
        max_dd = (bt["equity_strategy"] / bt["equity_strategy"].cummax() - 1).min()
        exposure = bt["signal"].mean()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Strategy return", f"{total_ret:.1%}")
        k2.metric("Buy & hold", f"{bh_ret:.1%}")
        k3.metric("Max drawdown", f"{max_dd:.1%}")
        k4.metric("Exposure", f"{exposure:.1%}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=bt["date"], y=bt["equity_strategy"], name="Strategy", line=dict(width=2.4)))
        fig.add_trace(go.Scatter(x=bt["date"], y=bt["equity_buyhold"], name="Buy & Hold", line=dict(width=2.0)))
        fig.update_layout(height=420, template="plotly_dark", title="Equity Curve", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.6, 0.4])
        fig2.add_trace(go.Scatter(x=bt["date"], y=bt["rsp_close"], name="RSP", line=dict(width=1.8)), row=1, col=1)
        fig2.add_trace(go.Scatter(x=bt["date"], y=bt["osc"], name="Breadth Osc", line=dict(width=1.8)), row=2, col=1)
        if "canary_comp" in bt.columns:
            fig2.add_trace(go.Scatter(x=bt["date"], y=bt["canary_comp"], name="Canary Composite", line=dict(width=1.5)), row=2, col=1)
        fig2.update_layout(height=500, template="plotly_dark", title="RSP + Sweet Spot Oscillator", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### Historical Score Series")
        show_cols = ["date", "bounce_score", "repair_score", "regime_score", "weekly_score", "master_score", "canary_comp", "canary_conf"]
        st.dataframe(score_df[show_cols].tail(40).round(2), use_container_width=True, hide_index=True)

with tab3:
    st.markdown("### Empirical Sweet Spot Summary")
    if summary_df.empty:
        st.warning("Sweet-spot summary not available.")
    else:
        st.dataframe(summary_df.round(3), use_container_width=True, hide_index=True)
    st.markdown("### Best Combo Zones")
    if combo_df.empty:
        st.info("No combo table available.")
    else:
        show = combo_df.copy()
        if "Features" in show.columns:
            show["Features"] = show["Features"].astype(str)
        st.dataframe(show.round(3), use_container_width=True, hide_index=True)
    st.markdown("### Upload History")
    hist = load_upload_history()
    if hist.empty:
        st.info("No realtime snapshot uploads recorded yet.")
    else:
        st.dataframe(hist.tail(25), use_container_width=True, hide_index=True)
