#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Holistic Oscillator Lab v1
- Research-first Streamlit app
- Searches for oscillator sweet spots across holistic breadth/leadership/risk buckets
- Ranks models versus SPY / RSP buy-and-hold
"""

from __future__ import annotations

import io
import itertools
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


# ---------------------------------------------------------
# App config / style
# ---------------------------------------------------------
st.set_page_config(
    page_title="Holistic Oscillator Lab v1",
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
.block-container{max-width:1550px;padding-top:1rem;padding-bottom:2rem;}
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
      <div style='font-size:1.8rem;font-weight:950;'>📈 Holistic Oscillator Lab v1</div>
      <div class='muted'>Search oscillator sweet spots across breadth, leadership, and risk buckets. Rank models by backtest results versus buy-and-hold.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

APP_DIR = Path("holistic_oscillator_lab_store")
APP_DIR.mkdir(exist_ok=True)
MODEL_PATH = APP_DIR / "best_model.json"
PARTIAL_RESULTS_PATH = APP_DIR / "partial_results.csv"
BEST_PREVIEW_PATH = APP_DIR / "best_preview.parquet"
BEST_EQUITY_PATH = APP_DIR / "best_equity.parquet"

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

BUCKETS = {
    "breadth": ["$BPSPX", "$BPNYA", "$SPXA50R", "$OEXA50R", "$OEXA150R", "$OEXA200R", "$NYMO", "$NYSI", "$NYHL", "$NYAD", "$SPXADP"],
    "leadership": ["RSP", "SPY", "RSP:SPY", "SMH:SPY", "IWM:SPY", "XLF:SPY", "HYG:IEF", "HYG:TLT"],
    "risk": ["$VIX", "VXX", "$TRIN", "$CPCE", "SPXS:SVOL"],
}
INVERSE = {"$VIX", "VXX", "$TRIN", "$CPCE", "SPXS:SVOL"}

DEFAULT_COMPONENT_WEIGHTS = {
    "$BPSPX": 1.25, "$BPNYA": 1.00, "$SPXA50R": 1.25, "$OEXA50R": 0.75, "$OEXA150R": 0.70, "$OEXA200R": 0.75,
    "$NYMO": 1.15, "$NYSI": 1.00, "$NYHL": 0.85, "$NYAD": 0.90, "$SPXADP": 0.90,
    "RSP": 0.85, "SPY": 0.60, "RSP:SPY": 1.00, "SMH:SPY": 0.90, "IWM:SPY": 0.75, "XLF:SPY": 0.70, "HYG:IEF": 0.95, "HYG:TLT": 0.70,
    "$VIX": 1.00, "VXX": 0.90, "$TRIN": 0.80, "$CPCE": 0.75, "SPXS:SVOL": 1.00,
}

OSCILLATOR_CHOICES = ["TSI", "RSI", "CCI", "ROC", "BB%"]
TRIGGER_CHOICES = ["signal_cross", "bull_cross_below_zero", "zero_cross"]
EXIT_CHOICES = ["signal_cross_down", "zero_cross_down", "bear_cross_above_zero"]

# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# Parsing uploads
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
            {
                "date": dt,
                "open": vals[0],
                "high": vals[1],
                "low": vals[2],
                "close": vals[3],
                "volume": vals[4],
            }
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
    high = pd.to_numeric(high, errors="coerce")
    low = pd.to_numeric(low, errors="coerce")
    close = pd.to_numeric(close, errors="coerce")
    tp = (high + low + close) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))

def percent_b(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    denom = (upper - lower).replace(0, np.nan)
    return (series - lower) / denom

def roc(series: pd.Series, length: int = 10) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return 100 * (s / s.shift(length) - 1)

def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    tp = (pd.to_numeric(high, errors="coerce") + pd.to_numeric(low, errors="coerce") + pd.to_numeric(close, errors="coerce")) / 3
    mf = tp * pd.to_numeric(volume, errors="coerce")
    pos = pd.Series(np.where(tp > tp.shift(1), mf, 0.0), index=tp.index)
    neg = pd.Series(np.where(tp < tp.shift(1), mf, 0.0), index=tp.index)
    pos_sum = pos.rolling(period).sum()
    neg_sum = neg.rolling(period).sum()
    ratio = pos_sum / neg_sum.replace(0, np.nan)
    return 100 - (100 / (1 + ratio))

def cmf(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    h = pd.to_numeric(high, errors="coerce")
    l = pd.to_numeric(low, errors="coerce")
    c = pd.to_numeric(close, errors="coerce")
    v = pd.to_numeric(volume, errors="coerce")
    denom = (h - l).replace(0, np.nan)
    mfm = ((c - l) - (h - c)) / denom
    mfv = mfm * v
    return mfv.rolling(period).sum() / v.rolling(period).sum().replace(0, np.nan)

# ---------------------------------------------------------
# Build normalized oscillator series by symbol
# ---------------------------------------------------------
def robust_z(series: pd.Series, window: int = 126) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mean = s.rolling(window, min_periods=max(20, window // 3)).mean()
    std = s.rolling(window, min_periods=max(20, window // 3)).std().replace(0, np.nan)
    z = (s - mean) / std
    return z.clip(-4, 4)

def oscillator_by_family(g: pd.DataFrame, family: str, params: Dict[str, Any]) -> Tuple[pd.Series, pd.Series]:
    c = g["close"]
    h = g.get("high", c)
    l = g.get("low", c)
    v = g.get("volume", pd.Series(index=g.index, dtype=float))
    family = family.upper()

    if family == "TSI":
        tsi, signal = true_strength_index(c, int(params["long"]), int(params["short"]), int(params["signal"]))
        return tsi, signal
    if family == "RSI":
        base = rsi(c, int(params["length"]))
        signal = ema(base, int(params.get("signal", 5)))
        return base, signal
    if family == "CCI":
        base = cci(h, l, c, int(params["length"]))
        signal = ema(base, int(params.get("signal", 5)))
        return base, signal
    if family == "ROC":
        base = roc(c, int(params["length"]))
        signal = ema(base, int(params.get("signal", 5)))
        return base, signal
    if family == "BB%":
        base = percent_b(c, int(params["length"]), float(params.get("std", 2.0)))
        base = (base - 0.5) * 100
        signal = ema(base, int(params.get("signal", 5)))
        return base, signal
    if family == "MFI":
        base = mfi(h, l, c, v, int(params["length"]))
        signal = ema(base, int(params.get("signal", 5)))
        return base, signal
    if family == "CMF":
        base = cmf(h, l, c, v, int(params["length"]))
        signal = ema(base, int(params.get("signal", 5)))
        return base, signal
    raise ValueError(f"Unsupported family: {family}")

@st.cache_data(show_spinner=False)
def build_symbol_oscillators(hist: pd.DataFrame, family: str, params_json: str) -> pd.DataFrame:
    params = json.loads(params_json)
    out = []
    for sym, g in hist.groupby("symbol", sort=False):
        g = g.sort_values("date").copy()
        osc, sig = oscillator_by_family(g, family, params)
        osc = robust_z(osc, 126)
        sig = robust_z(sig, 126)
        if sym in INVERSE:
            osc = -osc
            sig = -sig
        out.append(
            pd.DataFrame(
                {
                    "date": g["date"].values,
                    "symbol": sym,
                    "osc": osc.values,
                    "signal": sig.values,
                }
            )
        )
    return pd.concat(out, ignore_index=True)

# ---------------------------------------------------------
# Composite builder
# ---------------------------------------------------------
def available_bucket_members(symbols: List[str], bucket_name: str) -> List[str]:
    return [s for s in BUCKETS[bucket_name] if s in symbols]

def weighted_bucket_series(osc_df: pd.DataFrame, bucket_name: str, members: List[str], use_component_weights: bool = True) -> pd.DataFrame:
    if not members:
        return pd.DataFrame(columns=["date", "bucket", "bucket_osc", "bucket_signal"])
    sub = osc_df[osc_df["symbol"].isin(members)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["date", "bucket", "bucket_osc", "bucket_signal"])
    sub["wt"] = sub["symbol"].map(DEFAULT_COMPONENT_WEIGHTS).fillna(1.0 if not use_component_weights else 1.0)
    agg = (
        sub.groupby("date")
        .apply(
            lambda x: pd.Series(
                {
                    "bucket_osc": np.average(x["osc"], weights=x["wt"]),
                    "bucket_signal": np.average(x["signal"], weights=x["wt"]),
                    "members": int(x["symbol"].nunique()),
                }
            )
        )
        .reset_index()
    )
    agg["bucket"] = bucket_name
    return agg

def build_holistic_composite(
    osc_df: pd.DataFrame,
    bucket_weights: Dict[str, float],
    use_component_weights: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    symbols = osc_df["symbol"].dropna().unique().tolist()
    bucket_frames = []
    for bucket in BUCKETS:
        members = available_bucket_members(symbols, bucket)
        bf = weighted_bucket_series(osc_df, bucket, members, use_component_weights=use_component_weights)
        bucket_frames.append(bf)
    bucket_hist = pd.concat(bucket_frames, ignore_index=True) if bucket_frames else pd.DataFrame()

    if bucket_hist.empty:
        return pd.DataFrame(), pd.DataFrame()

    piv_osc = bucket_hist.pivot(index="date", columns="bucket", values="bucket_osc")
    piv_sig = bucket_hist.pivot(index="date", columns="bucket", values="bucket_signal")
    common = piv_osc.index.intersection(piv_sig.index)
    piv_osc = piv_osc.loc[common].copy()
    piv_sig = piv_sig.loc[common].copy()

    # Normalize weights to available columns
    active = [b for b in bucket_weights if b in piv_osc.columns]
    if not active:
        return pd.DataFrame(), bucket_hist
    w = pd.Series({b: bucket_weights[b] for b in active}, dtype=float)
    w = w / w.sum()

    holistic = pd.DataFrame(index=common)
    holistic["holistic_osc"] = (piv_osc[active] * w).sum(axis=1)
    holistic["holistic_signal"] = (piv_sig[active] * w).sum(axis=1)
    holistic["slope3"] = holistic["holistic_osc"].diff(3)
    holistic["gap"] = holistic["holistic_osc"] - holistic["holistic_signal"]
    holistic = holistic.reset_index()
    return holistic, bucket_hist

# ---------------------------------------------------------
# Backtest logic
# ---------------------------------------------------------
def compute_positions(
    osc: pd.Series,
    sig: pd.Series,
    trigger_rule: str,
    exit_rule: str,
) -> pd.Series:
    osc = pd.to_numeric(osc, errors="coerce")
    sig = pd.to_numeric(sig, errors="coerce")
    up_signal = (osc > sig) & (osc.shift(1) <= sig.shift(1))
    down_signal = (osc < sig) & (osc.shift(1) >= sig.shift(1))
    up_zero = (osc > 0) & (osc.shift(1) <= 0)
    down_zero = (osc < 0) & (osc.shift(1) >= 0)
    bull_below_zero = up_signal & (osc < 0)
    bear_above_zero = down_signal & (osc > 0)

    entry = {
        "signal_cross": up_signal,
        "bull_cross_below_zero": bull_below_zero,
        "zero_cross": up_zero,
    }[trigger_rule]

    exit_sig = {
        "signal_cross_down": down_signal,
        "zero_cross_down": down_zero,
        "bear_cross_above_zero": bear_above_zero,
    }[exit_rule]

    pos = pd.Series(0.0, index=osc.index)
    in_pos = False
    for i in range(len(pos)):
        if not in_pos and bool(entry.iloc[i]):
            in_pos = True
        elif in_pos and bool(exit_sig.iloc[i]):
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

    if len(eq) < 10:
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
    lead_bars = []
    for en, ex in zip(entries, exits):
        if ex <= en:
            continue
        tr = float(px.loc[ex] / px.loc[en] - 1)
        trade_rets.append(tr)
        # crude lead measurement: bars until benchmark confirms positive 5-bar momentum
        sub = px.loc[en:ex]
        conf = sub.pct_change(5)
        idx = conf[conf > 0].index
        if len(idx):
            lead_bars.append(int(sub.index.get_loc(idx[0])))

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
        "avg_lead_bars": float(np.mean(lead_bars)) if lead_bars else np.nan,
        "pos": pos,
        "strat_ret": strat_ret,
    }

def score_model(stats: Dict[str, Any]) -> float:
    if not stats:
        return -999.0
    sharpe = stats.get("sharpe", np.nan)
    ret = stats.get("strategy_return", np.nan)
    dd = abs(stats.get("max_dd", np.nan))
    win = stats.get("win_rate", np.nan)
    trades = stats.get("trades", 0)
    if pd.isna(sharpe) or pd.isna(ret) or pd.isna(dd):
        return -999.0
    penalty = 0.0
    if trades < 4:
        penalty += 0.30
    return float((1.8 * sharpe) + (1.2 * ret) - (1.0 * dd) + (0.4 * (0 if pd.isna(win) else win)) - penalty)

# ---------------------------------------------------------
# Search grid
# ---------------------------------------------------------
def family_param_grid(family: str) -> List[Dict[str, Any]]:
    f = family.upper()
    if f == "TSI":
        return [
            {"long": 20, "short": 10, "signal": 5},
            {"long": 25, "short": 13, "signal": 7},
            {"long": 30, "short": 15, "signal": 7},
            {"long": 35, "short": 15, "signal": 7},
        ]
    if f == "RSI":
        return [{"length": x, "signal": 5} for x in [7, 10, 14, 21]]
    if f == "CCI":
        return [{"length": x, "signal": 5} for x in [10, 14, 20, 30]]
    if f == "ROC":
        return [{"length": x, "signal": 5} for x in [5, 7, 10, 14]]
    if f == "BB%":
        return [{"length": x, "std": s, "signal": 5} for x in [10, 20] for s in [2.0, 2.5]]
    return []

def bucket_weight_grid() -> List[Dict[str, float]]:
    combos = []
    vals = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    for b in vals:
        for l in vals:
            r = round(1.0 - b - l, 2)
            if r < 0.10 or r > 0.40:
                continue
            combos.append({"breadth": b, "leadership": l, "risk": r})
    # Deduplicate rounding artifacts
    uniq = []
    seen = set()
    for x in combos:
        key = tuple(round(x[k], 2) for k in ["breadth", "leadership", "risk"])
        if abs(sum(key) - 1.0) < 1e-6 and key not in seen:
            seen.add(key)
            uniq.append(x)
    return uniq

@dataclass
class SearchConfig:
    benchmark: str
    oscillator_families: List[str]
    trigger_rules: List[str]
    exit_rules: List[str]
    top_n: int = 20
    use_component_weights: bool = True
    max_models: int = 500
    save_every: int = 20


def run_search(hist: pd.DataFrame, config: SearchConfig) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    piv_close = hist.pivot(index="date", columns="symbol", values="close").sort_index()
    if config.benchmark not in piv_close.columns:
        raise ValueError(f"Benchmark {config.benchmark} not found in uploaded history.")
    benchmark_price = piv_close[config.benchmark].dropna()

    results = []
    best_payload = {}
    preview_plot = pd.DataFrame()

    grids = []
    for fam in config.oscillator_families:
        for params in family_param_grid(fam):
            grids.append((fam, params))
    bw_grid = bucket_weight_grid()

    total_models = len(grids) * len(bw_grid) * len(config.trigger_rules) * len(config.exit_rules)
    if total_models > config.max_models:
        step = int(math.ceil(total_models / config.max_models))
        bw_grid = bw_grid[::step]
        total_models = len(grids) * len(bw_grid) * len(config.trigger_rules) * len(config.exit_rules)

    progress = st.progress(0.0, text="Searching oscillator sweet spots...")
    status = st.empty()
    live_table = st.empty()
    done = 0

    def flush_partial():
        if results:
            partial = pd.DataFrame(results).sort_values(["score", "sharpe", "alpha_return"], ascending=False).reset_index(drop=True)
            partial.to_csv(PARTIAL_RESULTS_PATH, index=False)
            live_table.dataframe(partial.head(config.top_n), width="stretch", hide_index=True)
        if best_payload:
            try:
                pd.DataFrame(best_payload["holistic_hist"]).to_parquet(BEST_PREVIEW_PATH, index=False)
                eq_df = pd.DataFrame({
                    "date": best_payload["equity"].index,
                    "strategy": best_payload["equity"].values,
                    "buyhold": best_payload["benchmark_equity"].values,
                })
                eq_df.to_parquet(BEST_EQUITY_PATH, index=False)
            except Exception:
                pass

    for fam, params in grids:
        osc_df = build_symbol_oscillators(hist, fam, json.dumps(params, sort_keys=True))
        for weights in bw_grid:
            holistic_hist, bucket_hist = build_holistic_composite(
                osc_df, weights, use_component_weights=config.use_component_weights
            )
            if holistic_hist.empty:
                done += len(config.trigger_rules) * len(config.exit_rules)
                progress.progress(min(done / max(total_models, 1), 1.0), text=f"Searching oscillator sweet spots... {done}/{total_models}")
                continue

            comp = holistic_hist.set_index("date").reindex(benchmark_price.index).dropna(subset=["holistic_osc", "holistic_signal"])
            if comp.empty:
                done += len(config.trigger_rules) * len(config.exit_rules)
                progress.progress(min(done / max(total_models, 1), 1.0), text=f"Searching oscillator sweet spots... {done}/{total_models}")
                continue

            price = benchmark_price.reindex(comp.index).dropna()
            comp = comp.reindex(price.index).dropna()

            for trig in config.trigger_rules:
                for ex in config.exit_rules:
                    pos = compute_positions(comp["holistic_osc"], comp["holistic_signal"], trig, ex)
                    stats = backtest_long_cash(price, pos)
                    if stats:
                        model_score = score_model(stats)
                        row = {
                            "score": model_score,
                            "family": fam,
                            "params": json.dumps(params),
                            "bucket_weights": json.dumps(weights),
                            "trigger": trig,
                            "exit": ex,
                            "return": stats["strategy_return"],
                            "benchmark_return": stats["benchmark_return"],
                            "alpha_return": stats["strategy_return"] - stats["benchmark_return"],
                            "cagr": stats["strategy_cagr"],
                            "benchmark_cagr": stats["benchmark_cagr"],
                            "max_dd": stats["max_dd"],
                            "sharpe": stats["sharpe"],
                            "trades": stats["trades"],
                            "win_rate": stats["win_rate"],
                            "avg_trade": stats["avg_trade"],
                            "avg_lead_bars": stats["avg_lead_bars"],
                        }
                        results.append(row)

                        if not best_payload or model_score > best_payload.get("score", -999):
                            best_payload = {
                                **row,
                                "holistic_hist": comp.reset_index(),
                                "equity": stats["equity"],
                                "benchmark_equity": stats["benchmark_equity"],
                                "bucket_hist": bucket_hist,
                            }
                            preview_plot = comp.reset_index().copy()

                    done += 1
                    progress.progress(min(done / max(total_models, 1), 1.0), text=f"Searching oscillator sweet spots... {done}/{total_models}")
                    if done % max(config.save_every, 1) == 0:
                        status.info(f"Checkpoint saved at {done}/{total_models}.")
                        flush_partial()

    flush_partial()
    progress.empty()
    status.success(f"Search complete. Evaluated {done} model tests.")
    res_df = pd.DataFrame(results).sort_values(["score", "sharpe", "alpha_return"], ascending=False).reset_index(drop=True) if results else pd.DataFrame()
    return res_df.head(config.top_n), best_payload, preview_plot


# ---------------------------------------------------------
# Charts
# ---------------------------------------------------------
def plot_oscillator(df: pd.DataFrame, title: str, osc_col: str, sig_col: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df[osc_col], mode="lines", name="Oscillator", line=dict(width=3)))
    fig.add_trace(go.Scatter(x=df["date"], y=df[sig_col], mode="lines", name="Signal", line=dict(width=2)))
    fig.add_hline(y=0, line_width=1, opacity=0.35)
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=420,
        margin=dict(l=30, r=20, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis_title="Date",
        yaxis_title="Oscillator",
    )
    return fig

def plot_equity(eq: pd.Series, bh: pd.Series, title: str) -> go.Figure:
    df = pd.DataFrame({"Strategy": eq, "BuyHold": bh}).dropna().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["index"], y=df["Strategy"], mode="lines", name="Strategy", line=dict(width=3)))
    fig.add_trace(go.Scatter(x=df["index"], y=df["BuyHold"], mode="lines", name="Buy & Hold", line=dict(width=2)))
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=420,
        margin=dict(l=30, r=20, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis_title="Date",
        yaxis_title="Growth of $1",
    )
    return fig

# ---------------------------------------------------------
# Main app
# ---------------------------------------------------------
def main() -> None:
    with st.sidebar:
        st.markdown("### Configuration")
        zip_file = st.file_uploader("Historical ZIP", type=["zip"])
        benchmark = st.selectbox("Benchmark", ["RSP", "SPY"], index=0)
        families = st.multiselect("Oscillator families to search", OSCILLATOR_CHOICES, default=["TSI", "RSI", "CCI", "ROC", "BB%"])
        trig_rules = st.multiselect("Entry trigger rules", TRIGGER_CHOICES, default=["signal_cross", "bull_cross_below_zero", "zero_cross"])
        exit_rules = st.multiselect("Exit rules", EXIT_CHOICES, default=["signal_cross_down", "zero_cross_down"])
        top_n = st.slider("Top results to keep", 5, 50, 15, 1)
        max_models = st.slider("Search budget (max model tests)", 100, 3000, 200, 100)
        save_every = st.slider("Checkpoint every N tests", 5, 100, 20, 5)
        use_component_weights = st.toggle("Use custom component weights", value=True)
        run_btn = st.button("Run oscillator sweet-spot search", type="primary", width="stretch")

        best_saved = load_json(MODEL_PATH, {})
        if best_saved:
            st.caption("Saved best model is available from prior run.")
        if PARTIAL_RESULTS_PATH.exists():
            st.caption("Partial results from a prior run are available below.")

    if not zip_file:
        st.info("Upload your StockCharts ZIP to start the oscillator search.")
        return

    try:
        daily_df, weekly_df = parse_stockcharts_zip(zip_file.read())
    except Exception as e:
        st.error(f"Could not parse ZIP: {e}")
        return

    symbols = sorted(daily_df["symbol"].dropna().unique().tolist())
    have_benchmark = benchmark in symbols

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='kpi'><div class='kpi-title'>Daily symbols</div><div class='kpi-value'>{len(symbols)}</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='kpi'><div class='kpi-title'>Rows</div><div class='kpi-value'>{len(daily_df):,}</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='kpi'><div class='kpi-title'>Date range</div><div class='kpi-value' style='font-size:1.1rem'>{daily_df['date'].min().date()} → {daily_df['date'].max().date()}</div></div>", unsafe_allow_html=True)
    with c4:
        pill = "green" if have_benchmark else "red"
        text = "Present" if have_benchmark else "Missing"
        st.markdown(f"<div class='kpi'><div class='kpi-title'>Benchmark {benchmark}</div><div class='kpi-value' style='font-size:1.2rem'>{text}</div><div class='small'>{benchmark}</div></div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Research Lab", "Best Model", "Data Check"])

    with tab1:
        st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
        st.subheader("Oscillator sweet-spot search")
        st.markdown(
            "This searches oscillator family, oscillator parameters, bucket weights, and entry/exit triggers. "
            "The score favors higher Sharpe, better total return, lower drawdown, and enough trades."
        )
        if run_btn:
            try:
                cfg = SearchConfig(
                    benchmark=benchmark,
                    oscillator_families=families,
                    trigger_rules=trig_rules,
                    exit_rules=exit_rules,
                    top_n=top_n,
                    use_component_weights=use_component_weights,
                    max_models=max_models,
                    save_every=save_every,
                )
                res_df, best_payload, preview_plot = run_search(daily_df, cfg)
                if res_df.empty:
                    st.warning("No valid models were produced. Try fewer restrictions or a larger search budget.")
                else:
                    st.session_state["lab_results"] = res_df
                    st.session_state["best_payload"] = best_payload
                    serializable = {
                        k: v for k, v in best_payload.items()
                        if k not in {"holistic_hist", "equity", "benchmark_equity", "bucket_hist"}
                    }
                    save_json(MODEL_PATH, serializable)

                    st.success(f"Search complete. Ranked {len(res_df)} top models.")
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
            except Exception as e:
                st.exception(e)
        elif "lab_results" in st.session_state:
            st.dataframe(st.session_state["lab_results"], width="stretch", hide_index=True)
        elif PARTIAL_RESULTS_PATH.exists():
            st.warning("Showing partial results from the last checkpointed run.")
            st.dataframe(pd.read_csv(PARTIAL_RESULTS_PATH).head(top_n), width="stretch", hide_index=True)
        else:
            st.info("Click **Run oscillator sweet-spot search** to rank the best models.")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        payload = st.session_state.get("best_payload", {})
        if not payload and BEST_PREVIEW_PATH.exists() and BEST_EQUITY_PATH.exists() and MODEL_PATH.exists():
            payload = load_json(MODEL_PATH, {})
            try:
                payload["holistic_hist"] = pd.read_parquet(BEST_PREVIEW_PATH)
                eq_df = pd.read_parquet(BEST_EQUITY_PATH)
                payload["equity"] = pd.Series(eq_df["strategy"].values, index=pd.to_datetime(eq_df["date"]))
                payload["benchmark_equity"] = pd.Series(eq_df["buyhold"].values, index=pd.to_datetime(eq_df["date"]))
                payload["bucket_hist"] = pd.DataFrame()
            except Exception:
                payload = {}
        if not payload:
            st.info("Run the search first. Then the best model will appear here.")
        else:
            params = json.loads(payload["params"])
            weights = json.loads(payload["bucket_weights"])

            a, b, c, d, e = st.columns(5)
            cards = [
                ("Family", payload["family"]),
                ("Sharpe", fmt_num(payload["sharpe"], 2)),
                ("Return", f"{payload['return']*100:.1f}%"),
                ("Max DD", f"{payload['max_dd']*100:.1f}%"),
                ("Trades", f"{int(payload['trades'])}"),
            ]
            for col, (title, value) in zip([a, b, c, d, e], cards):
                with col:
                    st.markdown(f"<div class='kpi'><div class='kpi-title'>{title}</div><div class='kpi-value' style='font-size:1.25rem'>{value}</div></div>", unsafe_allow_html=True)

            st.markdown(
                f"<span class='pill blue'>Params: {params}</span>"
                f"<span class='pill green'>Weights: {weights}</span>"
                f"<span class='pill yellow'>Entry: {payload['trigger']}</span>"
                f"<span class='pill red'>Exit: {payload['exit']}</span>",
                unsafe_allow_html=True,
            )

            hist = payload["holistic_hist"].copy()
            eq = payload["equity"]
            bh = payload["benchmark_equity"]

            c1, c2 = st.columns([1.15, 1.0])
            with c1:
                st.plotly_chart(
                    plot_oscillator(hist, f"Best holistic oscillator vs signal — {payload['family']}", "holistic_osc", "holistic_signal"),
                    width="stretch",
                )
            with c2:
                st.plotly_chart(
                    plot_equity(eq, bh, f"{benchmark} strategy vs buy-and-hold"),
                    width="stretch",
                )

            with st.expander("Bucket contribution detail", expanded=False):
                bucket_hist = payload["bucket_hist"]
                if isinstance(bucket_hist, pd.DataFrame) and not bucket_hist.empty:
                    st.dataframe(bucket_hist.sort_values(["date", "bucket"]).tail(60), width="stretch", hide_index=True)
                else:
                    st.info("No bucket detail available.")

    with tab3:
        st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
        st.subheader("Data check")
        st.markdown("Use this to confirm the ZIP parsed the expected symbols.")
        piv = daily_df.pivot(index="date", columns="symbol", values="close").sort_index()
        latest = piv.tail(5).reset_index()
        st.dataframe(latest, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
