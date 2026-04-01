#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Holistic Breadth Oscillator Dashboard
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


st.set_page_config(page_title="Holistic Breadth Oscillator Dashboard", page_icon="📈", layout="wide")

CUSTOM_CSS = """
<style>
.block-container{max-width:1600px;padding-top:1rem;padding-bottom:2rem;}
.main-title{
  padding:1rem 1.2rem;border-radius:18px;
  background:linear-gradient(135deg, rgba(96,165,250,.18), rgba(34,197,94,.10));
  border:1px solid rgba(148,163,184,.20);margin-bottom:1rem;
}
.kpi{
  background:rgba(255,255,255,.03);
  border:1px solid rgba(148,163,184,.18);
  border-radius:16px;padding:.85rem 1rem;
}
.kpi-title{font-size:.86rem;color:#b6c4df;font-weight:800;}
.kpi-value{font-size:1.6rem;color:white;font-weight:950;line-height:1.1;}
.small{font-size:.82rem;color:#99aacd;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown(
    """
    <div class='main-title'>
      <div style='font-size:1.8rem;font-weight:950;'>📈 Holistic Breadth Oscillator Dashboard</div>
      <div class='small'>Weighted bucket composite on top, stacked oscillators below.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

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
INVERSE_SERIES = {"$VIX", "VXX", "$TRIN", "$CPCE", "SPXS:SVOL"}

BUCKETS = {
    "Breadth": ["$BPSPX", "$BPNYA", "$SPXA50R", "$OEXA50R", "$OEXA150R", "$OEXA200R", "$NYMO", "$NYSI", "$NYHL", "$NYAD", "$SPXADP"],
    "Leadership": ["RSP:SPY", "SMH:SPY", "IWM:SPY", "XLF:SPY", "HYG:IEF", "HYG:TLT"],
    "Risk": ["$VIX", "VXX", "$TRIN", "$CPCE", "SPXS:SVOL"],
    "Price": ["RSP", "SPY"],
}
DEFAULT_COMPONENT_WEIGHTS = {
    "$BPSPX": 1.20, "$BPNYA": 1.00, "$SPXA50R": 1.25, "$OEXA50R": 0.80, "$OEXA150R": 0.70, "$OEXA200R": 0.75,
    "$NYMO": 1.15, "$NYSI": 1.00, "$NYHL": 0.85, "$NYAD": 0.85, "$SPXADP": 0.85,
    "RSP:SPY": 1.00, "SMH:SPY": 0.90, "IWM:SPY": 0.80, "XLF:SPY": 0.70, "HYG:IEF": 0.95, "HYG:TLT": 0.70,
    "$VIX": 1.00, "VXX": 0.90, "$TRIN": 0.85, "$CPCE": 0.75, "SPXS:SVOL": 1.00,
    "RSP": 1.15, "SPY": 0.70,
}


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return np.nan


def fmt_num(v: Any, d: int = 2) -> str:
    if pd.isna(v):
        return "n/a"
    return f"{float(v):.{d}f}"


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
        rows.append({"date": dt, "open": vals[0], "high": vals[1], "low": vals[2], "close": vals[3], "volume": vals[4]})
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


def parse_stockcharts_zip(file_bytes: bytes) -> pd.DataFrame:
    daily = []
    with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
        names = [n for n in zf.namelist() if (not n.endswith("/")) and n.lower().endswith(".csv")]
        prog = st.progress(0.0, text="Parsing ZIP...")
        for i, name in enumerate(names, start=1):
            try:
                content = zf.read(name)
                df = parse_stockcharts_csv(content)
                sym, tf = symbol_from_filename(name)
                if tf == "daily":
                    df["symbol"] = sym
                    daily.append(df)
            except Exception:
                pass
            prog.progress(i / max(len(names), 1), text=f"Parsing ZIP... {i}/{len(names)}")
        prog.empty()
    if not daily:
        raise ValueError("No daily CSV files were parsed from ZIP.")
    return pd.concat(daily, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)


def normalize_series_for_composite(series: pd.Series, inverse: bool = False) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return pd.Series(index=s.index, dtype=float)
    first_valid = s.dropna().iloc[0]
    if pd.isna(first_valid) or abs(first_valid) < 1e-12:
        return pd.Series(index=s.index, dtype=float)
    if inverse:
        return 100.0 * first_valid / s.replace(0, np.nan)
    return 100.0 * s / first_valid


def weighted_average_df(df: pd.DataFrame, weight_map: Dict[str, float]) -> pd.Series:
    cols = [c for c in df.columns if c in weight_map]
    if not cols:
        return pd.Series(index=df.index, dtype=float)
    w = pd.Series({c: weight_map[c] for c in cols}, dtype=float)
    w = w / w.sum()
    return (df[cols] * w).sum(axis=1)


def ema(series: pd.Series, span: int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    delta = s.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1 / period, adjust=False).mean()
    ma_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def true_strength_index(series: pd.Series, long_len: int = 25, short_len: int = 13, signal_len: int = 7) -> Tuple[pd.Series, pd.Series]:
    s = pd.to_numeric(series, errors="coerce")
    m = s.diff()
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


def roc(series: pd.Series, length: int = 12) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return 100 * (s / s.shift(length) - 1)


def build_ohlc_from_close(close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    c = pd.to_numeric(close, errors="coerce")
    h = c.rolling(2).max().fillna(c)
    l = c.rolling(2).min().fillna(c)
    return h, l, c


def build_bucket_and_holistic_prices(hist: pd.DataFrame, bucket_weights: Dict[str, float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    piv = hist.pivot(index="date", columns="symbol", values="close").sort_index()
    transformed = {}
    for sym in piv.columns:
        transformed[sym] = normalize_series_for_composite(piv[sym], inverse=(sym in INVERSE_SERIES))
    rebased = pd.DataFrame(transformed, index=piv.index)

    bucket_prices = pd.DataFrame(index=rebased.index)
    for bucket_name, members in BUCKETS.items():
        members_present = [m for m in members if m in rebased.columns]
        if not members_present:
            continue
        weight_map = {m: DEFAULT_COMPONENT_WEIGHTS.get(m, 1.0) for m in members_present}
        bucket_prices[bucket_name] = weighted_average_df(rebased[members_present], weight_map)

    active = [b for b in bucket_weights if b in bucket_prices.columns and bucket_weights[b] > 0]
    holistic = pd.DataFrame(index=bucket_prices.index)
    if active:
        w = pd.Series({b: bucket_weights[b] for b in active}, dtype=float)
        w = w / w.sum()
        holistic["Holistic"] = (bucket_prices[active] * w).sum(axis=1)
    return bucket_prices, holistic


def apply_lookback(df: pd.DataFrame, lookback: str) -> pd.DataFrame:
    if df.empty:
        return df
    mapping = {"6M": 126, "1Y": 252, "2Y": 504, "3Y": 756, "5Y": 1260}
    if lookback == "MAX":
        return df.copy()
    return df.tail(mapping[lookback]).copy()


def make_price_chart(holistic: pd.DataFrame, buckets: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=holistic.index, y=holistic["Holistic"], mode="lines", name="Holistic Price", line=dict(width=4)))
    for col in buckets.columns:
        fig.add_trace(go.Scatter(x=buckets.index, y=buckets[col], mode="lines", name=col, line=dict(width=1.5), opacity=0.6))
    fig.update_layout(
        template="plotly_white",
        height=460,
        title="Holistic Price with Bucket Composites",
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title="Date",
        yaxis_title="Rebased Value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def make_stacked_oscillator_chart(price: pd.Series, rsi_len: int, cci_len: int, tsi_long: int, tsi_short: int, tsi_signal: int, bb_len: int, bb_std: float, roc_len: int, show_roc: bool) -> go.Figure:
    high, low, close = build_ohlc_from_close(price)
    rsi_s = rsi(close, rsi_len)
    cci_s = cci(high, low, close, cci_len)
    tsi_s, tsi_sig = true_strength_index(close, tsi_long, tsi_short, tsi_signal)
    bb_s = percent_b(close, bb_len, bb_std)
    roc_s = roc(close, roc_len)

    rows = 5 if show_roc else 4
    heights = [0.22, 0.19, 0.24, 0.19, 0.16] if show_roc else [0.26, 0.22, 0.28, 0.24]
    titles = ["RSI", "CCI", "TSI", "%B", "ROC"] if show_roc else ["RSI", "CCI", "TSI", "%B"]

    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=heights, subplot_titles=titles)
    fig.add_trace(go.Scatter(x=price.index, y=rsi_s, mode="lines", name=f"RSI({rsi_len})", line=dict(width=2)), row=1, col=1)
    fig.add_hline(y=70, row=1, col=1, line_width=1, opacity=0.35)
    fig.add_hline(y=50, row=1, col=1, line_width=1, opacity=0.25, line_dash="dash")
    fig.add_hline(y=30, row=1, col=1, line_width=1, opacity=0.35)

    fig.add_trace(go.Scatter(x=price.index, y=cci_s, mode="lines", name=f"CCI({cci_len})", line=dict(width=2)), row=2, col=1)
    fig.add_hline(y=100, row=2, col=1, line_width=1, opacity=0.35)
    fig.add_hline(y=0, row=2, col=1, line_width=1, opacity=0.25, line_dash="dash")
    fig.add_hline(y=-100, row=2, col=1, line_width=1, opacity=0.35)

    fig.add_trace(go.Scatter(x=price.index, y=tsi_s, mode="lines", name=f"TSI({tsi_long},{tsi_short},{tsi_signal})", line=dict(width=2.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=price.index, y=tsi_sig, mode="lines", name="TSI Signal", line=dict(width=2)), row=3, col=1)
    fig.add_hline(y=0, row=3, col=1, line_width=1, opacity=0.25, line_dash="dash")

    fig.add_trace(go.Scatter(x=price.index, y=bb_s, mode="lines", name=f"%B({bb_len},{bb_std})", line=dict(width=2)), row=4, col=1)
    fig.add_hline(y=1.0, row=4, col=1, line_width=1, opacity=0.35)
    fig.add_hline(y=0.5, row=4, col=1, line_width=1, opacity=0.25, line_dash="dash")
    fig.add_hline(y=0.0, row=4, col=1, line_width=1, opacity=0.35)

    if show_roc:
        fig.add_trace(go.Scatter(x=price.index, y=roc_s, mode="lines", name=f"ROC({roc_len})", line=dict(width=2)), row=5, col=1)
        fig.add_hline(y=0, row=5, col=1, line_width=1, opacity=0.25, line_dash="dash")

    fig.update_layout(
        template="plotly_white",
        height=1100 if show_roc else 920,
        margin=dict(l=20, r=20, t=60, b=20),
        title="Holistic Oscillator Stack",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
    )
    fig.update_xaxes(title_text="Date", row=rows, col=1)
    return fig


with st.sidebar:
    st.markdown("### Data")
    zip_file = st.file_uploader("Historical ZIP", type=["zip"])

    st.markdown("### Bucket Weights")
    breadth_w = st.slider("Breadth", 0.0, 1.0, 0.45, 0.05)
    leadership_w = st.slider("Leadership", 0.0, 1.0, 0.20, 0.05)
    risk_w = st.slider("Risk", 0.0, 1.0, 0.15, 0.05)
    price_w = st.slider("Price", 0.0, 1.0, 0.20, 0.05)

    st.markdown("### Oscillator Parameters")
    lookback = st.selectbox("Chart Window", ["6M", "1Y", "2Y", "3Y", "5Y", "MAX"], index=1)
    rsi_len = st.slider("RSI Length", 5, 30, 14, 1)
    cci_len = st.slider("CCI Length", 5, 40, 20, 1)
    tsi_long = st.slider("TSI Long", 10, 50, 25, 1)
    tsi_short = st.slider("TSI Short", 4, 25, 13, 1)
    tsi_signal = st.slider("TSI Signal", 2, 15, 7, 1)
    bb_len = st.slider("BB% Length", 5, 40, 20, 1)
    bb_std = st.slider("BB% StdDev", 1.0, 3.0, 2.0, 0.1)
    roc_len = st.slider("ROC Length", 3, 30, 12, 1)
    show_roc = st.toggle("Show ROC panel", value=True)

if not zip_file:
    st.info("Upload your StockCharts ZIP to build the holistic price and oscillator stack.")
else:
    try:
        hist = parse_stockcharts_zip(zip_file.read())
        bucket_weights = {"Breadth": breadth_w, "Leadership": leadership_w, "Risk": risk_w, "Price": price_w}
        if sum(bucket_weights.values()) <= 0:
            st.error("At least one bucket weight must be greater than zero.")
            st.stop()

        bucket_prices, holistic = build_bucket_and_holistic_prices(hist, bucket_weights)
        if holistic.empty or "Holistic" not in holistic.columns:
            st.error("Could not build a holistic price series from the uploaded ZIP.")
            st.stop()

        bucket_prices = apply_lookback(bucket_prices, lookback)
        holistic = apply_lookback(holistic, lookback)
        holistic_price = holistic["Holistic"].dropna()

        latest = holistic_price.iloc[-1] if not holistic_price.empty else np.nan
        latest_rsi = rsi(holistic_price, rsi_len).iloc[-1] if len(holistic_price) else np.nan
        latest_cci = cci(*build_ohlc_from_close(holistic_price), cci_len).iloc[-1] if len(holistic_price) else np.nan
        latest_tsi, latest_tsi_sig = true_strength_index(holistic_price, tsi_long, tsi_short, tsi_signal)
        latest_bb = percent_b(holistic_price, bb_len, bb_std).iloc[-1] if len(holistic_price) else np.nan

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(f"<div class='kpi'><div class='kpi-title'>Holistic Price</div><div class='kpi-value'>{fmt_num(latest, 2)}</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='kpi'><div class='kpi-title'>RSI</div><div class='kpi-value'>{fmt_num(latest_rsi, 1)}</div></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='kpi'><div class='kpi-title'>CCI</div><div class='kpi-value'>{fmt_num(latest_cci, 1)}</div></div>", unsafe_allow_html=True)
        with c4:
            st.markdown(f"<div class='kpi'><div class='kpi-title'>TSI / Signal</div><div class='kpi-value'>{fmt_num(latest_tsi.iloc[-1],1)} / {fmt_num(latest_tsi_sig.iloc[-1],1)}</div></div>", unsafe_allow_html=True)
        with c5:
            st.markdown(f"<div class='kpi'><div class='kpi-title'>%B</div><div class='kpi-value'>{fmt_num(latest_bb, 2)}</div></div>", unsafe_allow_html=True)

        st.plotly_chart(make_price_chart(holistic, bucket_prices), width="stretch")
        st.plotly_chart(
            make_stacked_oscillator_chart(
                holistic_price, rsi_len, cci_len, tsi_long, tsi_short, tsi_signal, bb_len, bb_std, roc_len, show_roc
            ),
            width="stretch",
        )

        with st.expander("Bucket weights and latest bucket values", expanded=False):
            summary = pd.DataFrame({
                "Bucket": list(bucket_weights.keys()),
                "Weight": list(bucket_weights.values()),
                "Latest Value": [
                    bucket_prices[b].dropna().iloc[-1] if b in bucket_prices.columns and not bucket_prices[b].dropna().empty else np.nan
                    for b in bucket_weights.keys()
                ],
            })
            st.dataframe(summary, width="stretch", hide_index=True)

        with st.expander("Available symbols loaded from ZIP", expanded=False):
            sym_df = pd.DataFrame({"symbol": sorted(hist["symbol"].dropna().unique().tolist())})
            st.dataframe(sym_df, width="stretch", hide_index=True)
    except Exception as e:
        st.exception(e)
