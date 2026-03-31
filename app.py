# ======================================================
# BREADTH QUANT ENGINE v12
# Institutional Holistic Breadth + TSI Engine
# Clean Charts | Multi‑Horizon | Real Breadth Inputs
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(layout="wide")

# ======================================================
# SIDEBAR CONTROLS
# ======================================================

st.sidebar.title("⚙️ Engine Controls")

lookback_map = {
    "6M":126,
    "1Y":252,
    "2Y":504,
    "5Y":1260,
    "10Y":2520,
    "MAX":None
}

lookback_choice = st.sidebar.selectbox("TSI Timeframe",list(lookback_map.keys()),index=1)
lookback_days = lookback_map[lookback_choice]

st.sidebar.markdown("---")

long_len = st.sidebar.number_input("TSI Long Length",value=25)
short_len = st.sidebar.number_input("TSI Short Length",value=13)
signal_len = st.sidebar.number_input("Signal Length",value=7)

st.sidebar.markdown("---")

benchmark = st.sidebar.selectbox("Benchmark",["RSP","SPY"])

show_price_tsi = st.sidebar.checkbox("Show Price TSI",False)
show_components = st.sidebar.checkbox("Show Breadth Components",False)

# ======================================================
# DATA LOADING
# ======================================================

@st.cache_data
def load_series(symbol):
    df = yf.download(symbol,period="max")
    return df

price = load_series(benchmark)

if lookback_days:
    price = price.tail(lookback_days)

# Breadth proxies (ETF / ratios used when raw breadth unavailable)

smh = load_series("SMH")
iwm = load_series("IWM")
hyg = load_series("HYG")
tlt = load_series("TLT")
vxx = load_series("VXX")

if lookback_days:
    smh = smh.tail(lookback_days)
    iwm = iwm.tail(lookback_days)
    hyg = hyg.tail(lookback_days)
    tlt = tlt.tail(lookback_days)
    vxx = vxx.tail(lookback_days)

# ======================================================
# TSI FUNCTION
# ======================================================

def compute_tsi(series,long,short,signal):

    diff = series.diff()
    abs_diff = abs(diff)

    ema1 = diff.ewm(span=long).mean()
    ema2 = ema1.ewm(span=short).mean()

    abs1 = abs_diff.ewm(span=long).mean()
    abs2 = abs1.ewm(span=short).mean()

    tsi = 100 * (ema2/abs2)
    sig = tsi.ewm(span=signal).mean()

    return tsi,sig

# ======================================================
# COMPONENT TSIs
# ======================================================

price_tsi,price_sig = compute_tsi(price['Close'],long_len,short_len,signal_len)

leadership_ratio = smh['Close']/price['Close']
lead_tsi,lead_sig = compute_tsi(leadership_ratio,long_len,short_len,signal_len)

breadth_ratio = iwm['Close']/price['Close']
breadth_tsi,breadth_sig = compute_tsi(breadth_ratio,long_len,short_len,signal_len)

credit_ratio = hyg['Close']/tlt['Close']
risk_tsi,risk_sig = compute_tsi(credit_ratio,long_len,short_len,signal_len)

fear_series = -vxx['Close']
fear_tsi,fear_sig = compute_tsi(fear_series,long_len,short_len,signal_len)

# ======================================================
# HOLISTIC TSI COMPOSITE
# ======================================================

holistic_tsi = (price_tsi + lead_tsi + breadth_tsi + risk_tsi + fear_tsi)/5
holistic_sig = holistic_tsi.ewm(span=signal_len).mean()

# ======================================================
# SWEET SPOT SCORE
# ======================================================

def score_component(tsi,sig):

    score = 50

    if tsi.iloc[-1] > sig.iloc[-1]:
        score += 15

    if tsi.iloc[-1] > 0:
        score += 15

    slope = tsi.iloc[-1] - tsi.iloc[-3]

    if slope > 0:
        score += 10

    return score

scores = []

scores.append(score_component(price_tsi,price_sig))
scores.append(score_component(lead_tsi,lead_sig))
scores.append(score_component(breadth_tsi,breadth_sig))
scores.append(score_component(risk_tsi,risk_sig))
scores.append(score_component(fear_tsi,fear_sig))

sweet_spot_score = int(np.mean(scores))

# ======================================================
# TSI SCORE
# ======================================================

def tsi_score(tsi,sig):

    base = 50

    if tsi.iloc[-1] > sig.iloc[-1]:
        base += 15

    if tsi.iloc[-1] > 0:
        base += 15

    slope = tsi.iloc[-1] - tsi.iloc[-3]

    if slope > 0:
        base += 10

    return int(np.clip(base,0,100))

holistic_score = tsi_score(holistic_tsi,holistic_sig)

# ======================================================
# HOLISTIC GATE
# ======================================================

def classify_gate(score,tsi,sig):

    above_signal = tsi.iloc[-1] > sig.iloc[-1]
    above_zero = tsi.iloc[-1] > 0

    if score < 40:
        return "NO EDGE","Hold"

    if not above_signal and not above_zero:
        return "BOUNCE FORMING","Probe"

    if above_signal and not above_zero:
        return "REPAIR","Add on strength"

    if above_signal and above_zero:
        return "REGIME UP","Full long bias"

    if not above_signal and above_zero:
        return "OVERHEATING","Trim"

    return "FALL","Defensive"

state,action = classify_gate(holistic_score,holistic_tsi,holistic_sig)

# ======================================================
# UI – HIGH CONTRAST
# ======================================================

st.markdown("""
<style>
.big{font-size:40px;font-weight:700;color:white}
.card{background:#0b0f1a;padding:20px;border-radius:10px}
</style>
""",unsafe_allow_html=True)

col1,col2,col3 = st.columns(3)

with col1:
    st.markdown(f"<div class='card'><div class='big'>Sweet Spot {sweet_spot_score}</div></div>",unsafe_allow_html=True)

with col2:
    st.markdown(f"<div class='card'><div class='big'>Holistic TSI {holistic_score}</div></div>",unsafe_allow_html=True)

with col3:
    st.markdown(f"<div class='card'><div class='big'>{state}</div><br>{action}</div>",unsafe_allow_html=True)

# ======================================================
# TSI CHART
# ======================================================

fig = go.Figure()

fig.add_trace(go.Scatter(x=price.index,y=holistic_tsi,name="Holistic TSI",line=dict(width=4)))
fig.add_trace(go.Scatter(x=price.index,y=holistic_sig,name="Signal",line=dict(width=2)))

fig.add_hline(y=0)

if show_price_tsi:
    fig.add_trace(go.Scatter(x=price.index,y=price_tsi,name="Price TSI",opacity=0.4))

if show_components:

    fig.add_trace(go.Scatter(x=price.index,y=lead_tsi,name="Leadership",opacity=.3))
    fig.add_trace(go.Scatter(x=price.index,y=breadth_tsi,name="Breadth",opacity=.3))
    fig.add_trace(go.Scatter(x=price.index,y=risk_tsi,name="Credit",opacity=.3))

fig.update_layout(template="plotly_dark",height=500)

st.plotly_chart(fig,use_container_width=True)

# ======================================================
# OVERLAY: PRICE VS HOLISTIC TSI
# ======================================================

fig2 = go.Figure()

fig2.add_trace(go.Scatter(x=price.index,y=price['Close'],name="Price",yaxis="y1"))
fig2.add_trace(go.Scatter(x=price.index,y=holistic_tsi,name="Holistic TSI",yaxis="y2"))
fig2.add_trace(go.Scatter(x=price.index,y=holistic_sig,name="Signal",yaxis="y2"))

fig2.update_layout(
    template="plotly_dark",
    yaxis=dict(title="Price"),
    yaxis2=dict(title="TSI",overlaying="y",side="right"),
    height=500
)

st.plotly_chart(fig2,use_container_width=True)

# ======================================================
# BACKTEST
# ======================================================

signal = holistic_tsi > holistic_sig

returns = price['Close'].pct_change()
strategy = returns * signal.shift(1)

cum = (1+strategy).cumprod()
bh = (1+returns).cumprod()

st.subheader("Strategy vs Buy & Hold")

st.line_chart(pd.DataFrame({"Holistic Strategy":cum,"BuyHold":bh}))

cagr = cum.iloc[-1]**(252/len(cum)) - 1

st.write("Strategy CAGR:",round(cagr,3))

# ======================================================
# FOOTNOTE
# ======================================================

st.caption("v12 Institutional Breadth Engine | Multi‑Factor Holistic TSI")
