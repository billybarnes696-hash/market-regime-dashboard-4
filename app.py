#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Holistic Oscillator Lab v2
Phase 2:
- combo-aware
- percentile-aware
- regime-aware
- compares directly against RSP TSI baseline
- includes hybrid oscillator with price-confirmation bucket
"""

from __future__ import annotations

import io
import json
import math
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# ---------------------------------------------------------
# App config / style
# ---------------------------------------------------------
st.set_page_config(
    page_title="Holistic Oscillator Lab v2",
    layout="wide",
    page_icon="📈",
)

CUSTOM_CSS = """
<style>
:root{
  --bg:#07111f;--panel:#0d1a2f;--panel2:#101f39;--line:#23365d;
  --text:#f5f7fb;--muted:#a8b4cf;
  --green:#22c55e;--yellow:#f59e0b;--red:#ef4444;--blue:#60a5fa;
}
.block-container{max-width:1580px;padding-top:1rem;padding-bottom:2rem;}
.main-title{
  padding:1rem 1.2rem;border-radius:18px;
  background:linear-gradient(135deg, rgba(96,165,250,.18), rgba(34,197,94,.10));
  border:1px solid rgba(148,163,184,.20);margin-bottom:1rem;
}
.soft-card{
  background:linear-gradient(180deg, rgba(13,26,47,.98), rgba(9,18,33,.98));
  border:1px solid rgba(148,163,184,.18);border-radius:18px;padding:1rem;
  box-shadow:0 10px 30px rgba(0,0,0,.25);margin-bottom:1rem;
}
.kpi{
  background:rgba(255,255,255,.03);border:1px solid rgba(148,163,184,.18);
  border-radius:16px;padding:.85rem 1rem;
}
.kpi-title{font-size:.86rem;color:#b6c4df;font-weight:800;}
.kpi-value{font-size:2rem;color:white;font-weight:950;line-height:1.1;}
.muted{color:#99aacd;font-size:.88rem;}
.small{font-size:.82rem;color:#99aacd;}
.pill{display:inline-block;padding:.28rem .55rem;border-radius:999px;font-size:.82rem;font-weight:800;margin-right:.35rem;}
.green{background:rgba(34,197,94,.16);color:#bbf7d0;}
.yellow{background:rgba(245,158,11,.16);color:#fde68a;}
.red{background:rgba(239,68,68,.16);color:#fecaca;}
.blue{background:rgba(96,165,250,.16);color:#dbeafe;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown(
    """
    <div class='main-title'>
      <div style='font-size:1.8rem;font-weight:950;'>📈 Holistic Oscillator Lab v2</div>
      <div class='muted'>Phase 2 research lab: hybrid oscillator, bell-curve percentile map, forward-return sweet spots, and direct comparison versus RSP TSI.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

APP_DIR = Path("holistic_oscillator_lab_v2_store")
APP_DIR.mkdir(exist_ok=True)

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

INVERSE = {"$VIX", "VXX", "$TRIN", "$CPCE", "SPXS:SVOL"}

BUCKETS = {
    "breadth": ["$BPSPX", "$BPNYA", "$SPXA50R", "$OEXA50R", "$OEXA150R", "$OEXA200R", "$NYMO", "$NYSI", "$NYHL", "$NYAD", "$SPXADP"],
    "leadership": ["RSP:SPY", "SMH:SPY", "IWM:SPY", "XLF:SPY", "HYG:IEF", "HYG:TLT"],
    "risk": ["$VIX", "VXX", "$TRIN", "$CPCE", "SPXS:SVOL"],
    "price": ["RSP", "SPY"],
}

DEFAULT_COMPONENT_WEIGHTS = {
    "$BPSPX": 1.25, "$BPNYA": 1.00, "$SPXA50R": 1.25, "$OEXA50R": 0.80, "$OEXA150R": 0.70, "$OEXA200R": 0.75,
    "$NYMO": 1.20, "$NYSI": 1.05, "$NYHL": 0.90, "$NYAD": 0.90, "$SPXADP": 0.90,
    "RSP:SPY": 1.00, "SMH:SPY": 0.90, "IWM:SPY": 0.75, "XLF:SPY": 0.70, "HYG:IEF": 0.95, "HYG:TLT": 0.70,
    "$VIX": 1.00, "VXX": 0.90, "$TRIN": 0.80, "$CPCE": 0.75, "SPXS:SVOL": 1.00,
    "RSP": 1.15, "SPY": 0.65,
}

MODEL_LIBRARY = {
    "RSP_TSI_BASELINE": {
        "bucket_weights": {"price": 1.0},
        "combo": "TSI",
    },
    "HOLISTIC_TSI": {
        "bucket_weights": {"breadth": 0.60, "leadership": 0.20, "risk": 0.20},
        "combo": "TSI",
    },
    "HYBRID_TSI_PRICE": {
        "bucket_weights": {"breadth": 0.45, "leadership": 0.15, "risk": 0.15, "price": 0.25},
        "combo": "TSI",
    },
    "HOLISTIC_TSI_BB": {
        "bucket_weights": {"breadth": 0.50, "leadership": 0.15, "risk": 0.15, "price": 0.20},
        "combo": "TSI+BB",
    },
    "HOLISTIC_TSI_ROC": {
        "bucket_weights": {"breadth": 0.50, "leadership": 0.20, "risk": 0.10, "price": 0.20},
        "combo": "TSI+ROC",
    },
    "HOLISTIC_TSI_CCI": {
        "bucket_weights": {"breadth": 0.50, "leadership": 0.20, "risk": 0.10, "price": 0.20},
        "combo": "TSI+CCI",
    },
}

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return np.nan

def robust_z(series: pd.Series, window: int = 126) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mean = s.rolling(window, min_periods=max(20, window // 3)).mean()
    std = s.rolling(window, min_periods=max(20, window // 3)).std().replace(0, np.nan)
    z = (s - mean) / std
    return z.clip(-4, 4)

def fmt_num(v: Any, d: int = 2) -> str:
    if pd.isna(v):
        return "n/a"
    return f"{float(v):.{d}f}"

# ---------------------------------------------------------
# Parsing
# ---------------------------------------------------------
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
        vals = [safe_float(x) for x in parts[1:6]]
        rows.append(
            {"date": dt, "open": vals[0], "high": vals[1], "low": vals[2], "close": vals[3], "volume": vals[4]}
        )
    if not rows:
        raise ValueError("Could not parse rows from CSV.")
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
        names = [n for n in zf.namelist() if (not n.endswith("/")) and n.lower().endswith(".csv")]
        prog = st.progress(0.0, text="Parsing ZIP...")
        for i, name in enumerate(names, start=1):
            try:
                content = zf.read(name)
                df = parse_stockcharts_csv(content)
                sym, tf = symbol_from_filename(name)
                df["symbol"] = sym
                (weekly if tf == "weekly" else daily).append(df)
            except Exception:
                pass
            prog.progress(i / max(len(names), 1), text=f"Parsing ZIP... {i}/{len(names)}")
        prog.empty()

    if not daily:
        raise ValueError("No daily CSV files were parsed from ZIP.")
    daily_df = pd.concat(daily, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    weekly_df = pd.concat(weekly, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True) if weekly else pd.DataFrame(columns=daily_df.columns)
    return daily_df, weekly_df

# ---------------------------------------------------------
# Indicators
# ---------------------------------------------------------
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

def true_strength_index(series: pd.Series, long_len: int = 25, short_len: int = 13, signal_len: int = 7) -> Tuple[pd.Series, pd.Series]:
    series = pd.to_numeric(series, errors="coerce")
    m = series.diff()
    abs_m = m.abs()
    dsm = ema(ema(m, long_len), short_len)
    dsa = ema(ema(abs_m, long_len), short_len)
    tsi = 100 * (dsm / dsa.replace(0, np.nan))
    signal = ema(tsi, signal_len)
    return tsi, signal

def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    h = pd.to_numeric(high, errors="coerce")
    l = pd.to_numeric(low, errors="coerce")
    c = pd.to_numeric(close, errors="coerce")
    tp = (h + l + c) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))

def percent_b(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    ma = s.rolling(window).mean()
    std = s.rolling(window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    denom = (upper - lower).replace(0, np.nan)
    return (s - lower) / denom

def roc(series: pd.Series, length: int = 10) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return 100 * (s / s.shift(length) - 1)

# ---------------------------------------------------------
# Oscillator transforms
# ---------------------------------------------------------
def calc_family(g: pd.DataFrame, family: str, params: Dict[str, Any]) -> Tuple[pd.Series, pd.Series]:
    c = g["close"]
    h = g["high"] if "high" in g.columns else c
    l = g["low"] if "low" in g.columns else c
    family = family.upper()
    if family == "TSI":
        return true_strength_index(c, params["long"], params["short"], params["signal"])
    if family == "BB":
        base = (percent_b(c, params["length"], params["std"]) - 0.5) * 100
        return base, ema(base, params["signal"])
    if family == "ROC":
        base = roc(c, params["length"])
        return base, ema(base, params["signal"])
    if family == "CCI":
        base = cci(h, l, c, params["length"])
        return base, ema(base, params["signal"])
    raise ValueError(f"Unsupported family: {family}")

def build_symbol_family(hist: pd.DataFrame, family: str, params: Dict[str, Any]) -> pd.DataFrame:
    out = []
    for sym, g in hist.groupby("symbol", sort=False):
        g = g.sort_values("date").copy()
        val, sig = calc_family(g, family, params)
        val = robust_z(val, 126)
        sig = robust_z(sig, 126)
        if sym in INVERSE:
            val = -val
            sig = -sig
        out.append(pd.DataFrame({"date": g["date"].values, "symbol": sym, "val": val.values, "sig": sig.values}))
    return pd.concat(out, ignore_index=True)

def available_bucket_members(symbols: List[str], bucket_name: str) -> List[str]:
    return [s for s in BUCKETS[bucket_name] if s in symbols]

def weighted_bucket_series(family_df: pd.DataFrame, bucket_name: str, members: List[str]) -> pd.DataFrame:
    if not members:
        return pd.DataFrame(columns=["date", "bucket", "bucket_val", "bucket_sig"])
    sub = family_df[family_df["symbol"].isin(members)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["date", "bucket", "bucket_val", "bucket_sig"])
    sub["wt"] = sub["symbol"].map(DEFAULT_COMPONENT_WEIGHTS).fillna(1.0)
    agg = (
        sub.groupby("date")
        .apply(lambda x: pd.Series({
            "bucket_val": np.average(x["val"], weights=x["wt"]),
            "bucket_sig": np.average(x["sig"], weights=x["wt"]),
        }))
        .reset_index()
    )
    agg["bucket"] = bucket_name
    return agg

def build_combo_hist(
    hist: pd.DataFrame,
    bucket_weights: Dict[str, float],
    combo_name: str,
    params_map: Dict[str, Dict[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    symbols = sorted(hist["symbol"].dropna().unique().tolist())
    families = combo_name.split("+")
    combo_parts = []

    for fam in families:
        fam_df = build_symbol_family(hist, fam, params_map[fam])
        bucket_frames = []
        for bucket in BUCKETS:
            members = available_bucket_members(symbols, bucket)
            bf = weighted_bucket_series(fam_df, bucket, members)
            if not bf.empty:
                bf["family"] = fam
                bucket_frames.append(bf)
        fam_bucket_hist = pd.concat(bucket_frames, ignore_index=True) if bucket_frames else pd.DataFrame()
        if fam_bucket_hist.empty:
            continue
        piv_val = fam_bucket_hist.pivot(index="date", columns="bucket", values="bucket_val")
        piv_sig = fam_bucket_hist.pivot(index="date", columns="bucket", values="bucket_sig")
        active = [b for b in bucket_weights if b in piv_val.columns]
        if not active:
            continue
        w = pd.Series({b: bucket_weights[b] for b in active}, dtype=float)
        w = w / w.sum()
        comp = pd.DataFrame(index=piv_val.index.intersection(piv_sig.index))
        comp["osc"] = (piv_val.loc[comp.index, active] * w).sum(axis=1)
        comp["sig"] = (piv_sig.loc[comp.index, active] * w).sum(axis=1)
        comp["family"] = fam
        combo_parts.append(comp.reset_index())
    if not combo_parts:
        return pd.DataFrame(), pd.DataFrame()

    # Average across families if combo has multiple parts
    merged = pd.concat(combo_parts, ignore_index=True)
    final = (
        merged.groupby("date")
        .agg(osc=("osc", "mean"), sig=("sig", "mean"))
        .reset_index()
        .sort_values("date")
    )
    final["gap"] = final["osc"] - final["sig"]
    final["slope3"] = final["osc"].diff(3)
    final["pct_rank"] = final["osc"].expanding(min_periods=40).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    )
    final["slope_pct_rank"] = final["slope3"].expanding(min_periods=40).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    )
    final["state"] = final.apply(classify_state_row, axis=1)
    return final, merged

def classify_state_row(row: pd.Series) -> str:
    osc = safe_float(row.get("osc", np.nan))
    sig = safe_float(row.get("sig", np.nan))
    slope = safe_float(row.get("slope3", np.nan))
    pct = safe_float(row.get("pct_rank", np.nan))
    if pd.isna(osc) or pd.isna(sig):
        return "Unknown"
    if osc < 0 and osc < sig and slope <= 0 and pct <= 20:
        return "Regime Down"
    if osc < 0 and osc > sig and slope > 0 and pct <= 20:
        return "Bounce"
    if osc < 0 and osc > sig and slope > 0 and pct <= 45:
        return "Repair"
    if osc >= 0 and osc > sig and slope > 0 and pct < 85:
        return "Regime Up"
    if osc >= 0 and pct >= 85 and slope <= 0:
        return "Overheating"
    if osc >= 0 and osc < sig and slope < 0:
        return "Fall"
    return "Transitional"

# ---------------------------------------------------------
# Forward stats / backtests
# ---------------------------------------------------------
def signal_cross_up(osc: pd.Series, sig: pd.Series) -> pd.Series:
    return (osc > sig) & (osc.shift(1) <= sig.shift(1))

def signal_cross_down(osc: pd.Series, sig: pd.Series) -> pd.Series:
    return (osc < sig) & (osc.shift(1) >= sig.shift(1))

def zero_cross_up(osc: pd.Series) -> pd.Series:
    return (osc > 0) & (osc.shift(1) <= 0)

def zero_cross_down(osc: pd.Series) -> pd.Series:
    return (osc < 0) & (osc.shift(1) >= 0)

def build_positions(reg_df: pd.DataFrame, entry_mode: str, exit_mode: str, lower_pct: float, upper_pct: float) -> pd.Series:
    osc = reg_df["osc"]
    sig = reg_df["sig"]
    pct = reg_df["pct_rank"]
    slope_pct = reg_df["slope_pct_rank"]

    entry_cross = signal_cross_up(osc, sig)
    entry_zero = zero_cross_up(osc)
    exit_cross = signal_cross_down(osc, sig)
    exit_zero = zero_cross_down(osc)

    if entry_mode == "bull_cross_below_zero":
        entry = entry_cross & (osc < 0) & (pct <= lower_pct) & (slope_pct >= 55)
    elif entry_mode == "bull_cross_or_zero_from_washout":
        entry = ((entry_cross & (osc < 0)) | entry_zero) & (pct <= lower_pct) & (slope_pct >= 55)
    elif entry_mode == "repair_zone_turn":
        entry = (osc > sig) & (osc < 0) & (pct <= lower_pct) & (slope_pct >= 60)
    else:
        entry = entry_cross

    if exit_mode == "cross_from_positive_or_hot":
        exit_ = (exit_cross & (osc > 0)) | ((pct >= upper_pct) & (slope_pct <= 45))
    elif exit_mode == "zero_cross_down":
        exit_ = exit_zero | ((pct >= upper_pct) & (slope_pct <= 45))
    elif exit_mode == "hot_rollover":
        exit_ = (pct >= upper_pct) & (slope_pct <= 45)
    else:
        exit_ = exit_cross

    pos = pd.Series(0.0, index=reg_df.index)
    in_pos = False
    for i in range(len(pos)):
        if not in_pos and bool(entry.iloc[i]):
            in_pos = True
        elif in_pos and bool(exit_.iloc[i]):
            in_pos = False
        pos.iloc[i] = 1.0 if in_pos else 0.0
    return pos

def backtest_long_cash(price: pd.Series, pos: pd.Series) -> Dict[str, Any]:
    px = pd.to_numeric(price, errors="coerce").dropna()
    pos = pos.reindex(px.index).fillna(0.0)
    ret = px.pct_change().fillna(0.0)
    strat_ret = ret * pos.shift(1).fillna(0.0)
    eq = (1 + strat_ret).cumprod()
    bh = (1 + ret).cumprod()

    if len(eq) < 50:
        return {}

    years = max(len(eq) / 252.0, 1e-6)
    total_return = float(eq.iloc[-1] - 1)
    bh_return = float(bh.iloc[-1] - 1)
    cagr = float(eq.iloc[-1] ** (1 / years) - 1)
    bh_cagr = float(bh.iloc[-1] ** (1 / years) - 1)
    vol = float(strat_ret.std() * np.sqrt(252))
    sharpe = float((strat_ret.mean() * 252) / (vol if vol > 1e-9 else np.nan))
    dd = eq / eq.cummax() - 1
    max_dd = float(dd.min())

    changes = pos.diff().fillna(pos.iloc[0])
    entries = list(changes[changes > 0].index)
    exits = list(changes[changes < 0].index)
    if len(exits) < len(entries):
        exits.append(px.index[-1])

    trade_rets = []
    for en, ex in zip(entries, exits):
        if ex <= en:
            continue
        trade_rets.append(float(px.loc[ex] / px.loc[en] - 1))

    return {
        "equity": eq,
        "benchmark_equity": bh,
        "strategy_return": total_return,
        "benchmark_return": bh_return,
        "strategy_cagr": cagr,
        "benchmark_cagr": bh_cagr,
        "volatility": vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "trades": int(len(trade_rets)),
        "win_rate": float(np.mean([x > 0 for x in trade_rets])) if trade_rets else np.nan,
        "avg_trade": float(np.mean(trade_rets)) if trade_rets else np.nan,
        "pos": pos,
    }

def add_forward_returns(df: pd.DataFrame, price: pd.Series, horizons: List[int]) -> pd.DataFrame:
    out = df.copy()
    px = price.reindex(out["date"]).reset_index(drop=True)
    for h in horizons:
        out[f"fwd_{h}d"] = px.shift(-h) / px - 1
    return out

def heatmap_table(df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    bins = [0, 10, 20, 35, 50, 65, 80, 90, 100]
    labels = ["0-10", "10-20", "20-35", "35-50", "50-65", "65-80", "80-90", "90-100"]
    temp = df.copy()
    temp["zone"] = pd.cut(temp["pct_rank"], bins=bins, labels=labels, include_lowest=True)
    rows = []
    for zone, g in temp.groupby("zone", observed=True):
        row = {"Percentile Zone": zone}
        for h in horizons:
            row[f"{h}D Avg"] = g[f"fwd_{h}d"].mean()
            row[f"{h}D Med"] = g[f"fwd_{h}d"].median()
        rows.append(row)
    return pd.DataFrame(rows)

# ---------------------------------------------------------
# Charts
# ---------------------------------------------------------
def plot_oscillator(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["osc"], mode="lines", name="Oscillator", line=dict(width=3)))
    fig.add_trace(go.Scatter(x=df["date"], y=df["sig"], mode="lines", name="Signal", line=dict(width=2)))
    fig.add_hline(y=0, line_width=1, opacity=0.35)
    fig.update_layout(
        title=title, template="plotly_white", height=420,
        margin=dict(l=30, r=20, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis_title="Date", yaxis_title="Oscillator",
    )
    return fig

def plot_equity(eq: pd.Series, bh: pd.Series, title: str) -> go.Figure:
    df = pd.DataFrame({"Strategy": eq, "BuyHold": bh}).dropna().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["index"], y=df["Strategy"], mode="lines", name="Strategy", line=dict(width=3)))
    fig.add_trace(go.Scatter(x=df["index"], y=df["BuyHold"], mode="lines", name="Buy & Hold", line=dict(width=2)))
    fig.update_layout(
        title=title, template="plotly_white", height=420,
        margin=dict(l=30, r=20, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis_title="Date", yaxis_title="Growth of $1",
    )
    return fig

def plot_bell_curve(df: pd.DataFrame, current_value: float, title: str) -> go.Figure:
    hist = df["osc"].dropna()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=hist, histnorm="probability density", nbinsx=50, name="History"))
    fig.add_vline(x=current_value, line_width=3, annotation_text=f"Current {current_value:.2f}")
    fig.update_layout(
        title=title, template="plotly_white", height=380,
        margin=dict(l=30, r=20, t=50, b=30), xaxis_title="Oscillator", yaxis_title="Density",
        showlegend=False
    )
    return fig

def plot_heatmap(ht: pd.DataFrame, horizons: List[int]) -> go.Figure:
    z = ht[[f"{h}D Avg" for h in horizons]].to_numpy(dtype=float)
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=[f"{h}D Avg" for h in horizons],
        y=ht["Percentile Zone"],
        text=np.round(z * 100, 2),
        texttemplate="%{text}%",
        colorbar_title="Return",
    ))
    fig.update_layout(
        title="Forward returns by oscillator percentile zone",
        template="plotly_white",
        height=380,
        margin=dict(l=30, r=20, t=50, b=30),
        xaxis_title="Forward horizon",
        yaxis_title="Oscillator percentile zone",
    )
    return fig

# ---------------------------------------------------------
# Main app
# ---------------------------------------------------------
def main() -> None:
    with st.sidebar:
        st.markdown("### Inputs")
        zip_file = st.file_uploader("Historical ZIP", type=["zip"])
        phase1_file = st.file_uploader("Optional: Phase 1 top-results CSV", type=["csv"])
        benchmark = st.selectbox("Benchmark", ["RSP", "SPY"], index=0)
        entry_mode = st.selectbox(
            "Entry logic",
            ["bull_cross_below_zero", "bull_cross_or_zero_from_washout", "repair_zone_turn"],
            index=0,
        )
        exit_mode = st.selectbox(
            "Exit logic",
            ["cross_from_positive_or_hot", "zero_cross_down", "hot_rollover"],
            index=0,
        )
        lower_pct = st.slider("Lower percentile for entries", 5, 40, 25, 5)
        upper_pct = st.slider("Upper percentile for exits", 60, 95, 85, 5)
        lookback = st.selectbox("Chart window", ["6M", "1Y", "2Y", "3Y", "5Y", "MAX"], index=2)
        run_btn = st.button("Build Phase 2 models", type="primary", width="stretch")

    if not zip_file:
        st.info("Upload your historical StockCharts ZIP to start Phase 2.")
        return

    daily_df, weekly_df = parse_stockcharts_zip(zip_file.read())
    symbols = sorted(daily_df["symbol"].dropna().unique().tolist())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='kpi'><div class='kpi-title'>Daily symbols</div><div class='kpi-value'>{len(symbols)}</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='kpi'><div class='kpi-title'>Rows</div><div class='kpi-value'>{len(daily_df):,}</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='kpi'><div class='kpi-title'>Date range</div><div class='kpi-value' style='font-size:1.05rem'>{daily_df['date'].min().date()} → {daily_df['date'].max().date()}</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='kpi'><div class='kpi-title'>Benchmark</div><div class='kpi-value' style='font-size:1.25rem'>{benchmark}</div></div>", unsafe_allow_html=True)

    if phase1_file is not None:
        st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
        st.subheader("Imported Phase 1 winners")
        p1 = pd.read_csv(phase1_file)
        st.dataframe(p1.head(15), width="stretch", hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if not run_btn:
        st.info("Click **Build Phase 2 models** to compare the hybrid and combo oscillators.")
        return

    # Parameter map centered on your research
    params_map = {
        "TSI": {"long": 25, "short": 13, "signal": 7},
        "BB": {"length": 20, "std": 2.0, "signal": 5},
        "ROC": {"length": 10, "signal": 5},
        "CCI": {"length": 20, "signal": 5},
    }

    piv_close = daily_df.pivot(index="date", columns="symbol", values="close").sort_index()
    if benchmark not in piv_close.columns:
        st.error(f"Benchmark {benchmark} is missing from the history.")
        return
    benchmark_price = piv_close[benchmark].dropna()

    results = []
    model_payloads: Dict[str, Dict[str, Any]] = {}
    progress = st.progress(0.0, text="Building Phase 2 models...")
    items = list(MODEL_LIBRARY.items())

    for i, (model_name, spec) in enumerate(items, start=1):
        hist, raw_parts = build_combo_hist(daily_df, spec["bucket_weights"], spec["combo"], params_map)
        if hist.empty:
            continue
        comp = hist.set_index("date").reindex(benchmark_price.index).dropna(subset=["osc", "sig"])
        price = benchmark_price.reindex(comp.index).dropna()
        comp = comp.reindex(price.index).dropna()
        comp = comp.reset_index()

        pos = build_positions(comp, entry_mode, exit_mode, lower_pct, upper_pct)
        stats = backtest_long_cash(price, pos)

        comp["position"] = pos.values
        comp = add_forward_returns(comp, price, [5, 10, 20, 40])
        current = comp.iloc[-1]
        heat = heatmap_table(comp, [5, 10, 20, 40])

        row = {
            "model": model_name,
            "combo": spec["combo"],
            "bucket_weights": json.dumps(spec["bucket_weights"]),
            "return": stats.get("strategy_return", np.nan),
            "benchmark_return": stats.get("benchmark_return", np.nan),
            "alpha_return": stats.get("strategy_return", np.nan) - stats.get("benchmark_return", np.nan),
            "cagr": stats.get("strategy_cagr", np.nan),
            "benchmark_cagr": stats.get("benchmark_cagr", np.nan),
            "max_dd": stats.get("max_dd", np.nan),
            "sharpe": stats.get("sharpe", np.nan),
            "trades": stats.get("trades", np.nan),
            "win_rate": stats.get("win_rate", np.nan),
            "avg_trade": stats.get("avg_trade", np.nan),
            "current_osc": current["osc"],
            "current_sig": current["sig"],
            "current_pct": current["pct_rank"],
            "current_slope_pct": current["slope_pct_rank"],
            "current_state": current["state"],
        }
        results.append(row)
        model_payloads[model_name] = {
            "hist": comp,
            "equity": stats.get("equity", pd.Series(dtype=float)),
            "benchmark_equity": stats.get("benchmark_equity", pd.Series(dtype=float)),
            "heat": heat,
            "raw_parts": raw_parts,
        }
        progress.progress(i / len(items), text=f"Building Phase 2 models... {i}/{len(items)}")

    progress.empty()

    res_df = pd.DataFrame(results).sort_values(["alpha_return", "sharpe", "max_dd"], ascending=[False, False, False]).reset_index(drop=True)
    if res_df.empty:
        st.warning("No models were built.")
        return

    st.session_state["phase2_results"] = res_df
    st.session_state["phase2_payloads"] = model_payloads

    # Apply lookback
    lb_map = {"6M": 126, "1Y": 252, "2Y": 504, "3Y": 756, "5Y": 1260}
    tabs = st.tabs(["Leaderboard", "Best Model", "Bell Curve", "Forward Heatmap", "Diagnostics"])

    with tabs[0]:
        st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
        st.subheader("Phase 2 leaderboard")
        st.dataframe(
            res_df.assign(
                return_pct=lambda x: (x["return"] * 100).round(2),
                benchmark_return_pct=lambda x: (x["benchmark_return"] * 100).round(2),
                alpha_pct=lambda x: (x["alpha_return"] * 100).round(2),
                max_dd_pct=lambda x: (x["max_dd"] * 100).round(2),
                win_rate_pct=lambda x: (x["win_rate"] * 100).round(1),
            ),
            width="stretch",
            hide_index=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    best_name = res_df.iloc[0]["model"]
    best_payload = model_payloads[best_name]
    best_hist = best_payload["hist"].copy()
    if lookback != "MAX":
        n = lb_map[lookback]
        best_hist = best_hist.tail(n)

    with tabs[1]:
        st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
        st.subheader(f"Best model: {best_name}")
        meta = res_df.iloc[0]
        st.markdown(
            f"<span class='pill blue'>Combo: {meta['combo']}</span>"
            f"<span class='pill green'>State: {meta['current_state']}</span>"
            f"<span class='pill yellow'>Current Percentile: {meta['current_pct']:.1f}%</span>"
            f"<span class='pill red'>Current Slope Percentile: {meta['current_slope_pct']:.1f}%</span>",
            unsafe_allow_html=True,
        )
        a, b, c, d, e = st.columns(5)
        cards = [
            ("Return", f"{meta['return']*100:.1f}%"),
            ("Alpha vs BH", f"{meta['alpha_return']*100:.1f}%"),
            ("Sharpe", fmt_num(meta["sharpe"], 2)),
            ("Max DD", f"{meta['max_dd']*100:.1f}%"),
            ("Trades", f"{int(meta['trades'])}"),
        ]
        for col, (title, value) in zip([a, b, c, d, e], cards):
            with col:
                st.markdown(f"<div class='kpi'><div class='kpi-title'>{title}</div><div class='kpi-value' style='font-size:1.25rem'>{value}</div></div>", unsafe_allow_html=True)

        c1, c2 = st.columns([1.15, 1.0])
        with c1:
            st.plotly_chart(plot_oscillator(best_hist, f"{best_name} oscillator vs signal"), width="stretch")
        with c2:
            st.plotly_chart(plot_equity(best_payload["equity"], best_payload["benchmark_equity"], f"{benchmark}: strategy vs buy-and-hold"), width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[2]:
        st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
        st.subheader("Bell-curve regime map")
        current_val = safe_float(best_payload["hist"]["osc"].iloc[-1])
        st.plotly_chart(plot_bell_curve(best_payload["hist"], current_val, f"{best_name} historical oscillator distribution"), width="stretch")
        st.markdown(
            f"""
            **Current state:** {meta['current_state']}  
            **Current oscillator percentile:** {meta['current_pct']:.1f}%  
            **Interpretation:** lower percentiles are more washed-out / repair-like, while upper percentiles are more stretched / overheating-like.
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[3]:
        st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
        st.subheader("Forward-return sweet spot map")
        heat = best_payload["heat"]
        st.plotly_chart(plot_heatmap(heat, [5, 10, 20, 40]), width="stretch")
        st.dataframe(
            heat.assign(**{c: lambda x, col=c: (x[col] * 100).round(2) for c in heat.columns if c != "Percentile Zone"}),
            width="stretch",
            hide_index=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[4]:
        st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
        st.subheader("Diagnostics")
        st.markdown("Compare all model states and current locations.")
        diag = res_df[["model", "combo", "current_state", "current_pct", "current_slope_pct", "return", "alpha_return", "sharpe", "max_dd", "trades"]].copy()
        st.dataframe(
            diag.assign(
                return_pct=lambda x: (x["return"] * 100).round(2),
                alpha_pct=lambda x: (x["alpha_return"] * 100).round(2),
                max_dd_pct=lambda x: (x["max_dd"] * 100).round(2),
            ),
            width="stretch",
            hide_index=True,
        )

        with st.expander("Best model raw component history", expanded=False):
            raw_parts = best_payload["raw_parts"].copy()
            st.dataframe(raw_parts.tail(200), width="stretch", hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
