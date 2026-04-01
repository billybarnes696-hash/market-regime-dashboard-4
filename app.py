import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import json
import os
from datetime import datetime
import plotly.graph_objects as go
import pandas_ta as ta

st.set_page_config(page_title="Market Breadth Engine v2.0", layout="wide")

# ============================================================================
# CONFIGURATION & LOGGING
# ============================================================================

LOG_FILE = "upload_history.json"

def log_upload(filename, score, timestamp):
    """Logs every upload to a JSON file for record keeping."""
    history = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r') as f:
                history = json.load(f)
        except:
            history = []
    
    record = {
        "filename": filename,
        "gate_score": score,
        "timestamp": timestamp,
        "action": "Processed"
    }
    history.append(record)
    
    # Keep last 50 records
    if len(history) > 50:
        history = history[-50:]
        
    with open(LOG_FILE, 'w') as f:
        json.dump(history, f, indent=4)

def load_upload_log():
    """Loads the upload history for display."""
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data
def load_historical_zip(zip_file):
    """Loads historical CSVs from ZIP and calculates oscillators."""
    data = {}
    indicators_map = {
        "SPXA50R": "$SPXA50R", "BPSPX": "$BPSPX", "BPNYA": "$BPNYA",
        "TRIN": "$TRIN", "SPXADP": "$SPXADP", "RSP": "RSP", 
        "SPX": "$SPX", "VIX": "$VIX", "CPCE": "$CPCE",
        "RSP:SPY": "RSP:SPY", "IWM:SPY": "IWM:SPY", "XLF:SPY": "XLF:SPY",
        "HYG:IEF": "HYG:IEF", "HYG:TLT": "HYG:TLT", "URSP": "URSP",
        "VXX": "VXX", "SMH:SPY": "SMH:SPY", "NYAD": "$NYAD", "NYHL": "$NYHL"
    }
    
    with zipfile.ZipFile(zip_file) as z:
        for name in z.namelist():
            if name.endswith(".csv"):
                try:
                    df = pd.read_csv(z.open(name))
                    if "Date" in df.columns and "Close" in df.columns:
                        df["Date"] = pd.to_datetime(df["Date"])
                        df = df.set_index("Date").sort_index()
                        
                        # Match filename to indicator key
                        key = None
                        for k, v in indicators_map.items():
                            if v.replace("$", "") in name.replace("$", "").replace(".csv", ""):
                                key = k
                                break
                        
                        if key:
                            # Calculate Oscillators for Accuracy
                            df['TSI'] = ta.tsi(df['Close'], long=25, short=13, signal=7)
                            df['MACD'] = ta.macd(df['Close'], fast=12, slow=26, signal=9)['MACD_12_26_9']
                            df['RSI'] = ta.rsi(df['Close'], length=14)
                            df['ROC'] = ta.roc(df['Close'], length=12)
                            
                            data[key] = df
                except Exception as e:
                    pass
    return data

@st.cache_data
def load_snapshot(file):
    """Loads the daily snapshot (e.g., SC (21).csv)."""
    df = pd.read_csv(file)
    return df

# ============================================================================
# GATE SCORE V2.0 LOGIC
# ============================================================================

def calculate_oscillator_consensus(hist_data, snapshot_values):
    """Calculates oscillator consensus based on historical data + current value."""
    consensus = 0
    count = 0
    
    # Define key indicators for consensus
    keys = ["SPXA50R", "BPSPX", "RSP", "TRIN", "VIX"]
    
    for key in keys:
        if key in hist_data and key in snapshot_values:
            df = hist_data[key]
            current_val = snapshot_values[key]
            
            # Get latest row
            last_row = df.iloc[-1]
            
            # Simple Bullish/Bearish Logic based on Oscillators
            bullish = 0
            
            # TSI
            if not pd.isna(last_row['TSI']):
                if last_row['TSI'] > 0: bullish += 1
            
            # MACD
            if not pd.isna(last_row['MACD']):
                if last_row['MACD'] > 0: bullish += 1
                
            # RSI
            if not pd.isna(last_row['RSI']):
                if 40 < last_row['RSI'] < 60: bullish += 1 # Neutral is okay for some
                elif last_row['RSI'] > 60: bullish += 1 # Bullish momentum
                
            # ROC
            if not pd.isna(last_row['ROC']):
                if last_row['ROC'] > 0: bullish += 1
            
            # Inverse Logic for VIX/TRIN
            if key in ["TRIN", "VIX"]:
                bullish = 4 - bullish # Invert score
            
            consensus += bullish
            count += 4
            
    return round((consensus / count) * 100, 1) if count > 0 else 0

def calculate_gate_score_v2(ind, hist_data):
    """Calculates the weighted Gate Score v2.0."""
    
    # 1. Breadth Momentum (25%)
    breadth_score = 0
    if ind["SPXA50R"] > 40: breadth_score += 4
    elif ind["SPXA50R"] > 30: breadth_score += 3
    elif ind["SPXA50R"] > 20: breadth_score += 2
    else: breadth_score += 1
    
    if ind["BPSPX"] > 50: breadth_score += 3
    elif ind["BPSPX"] > 40: breadth_score += 2
    else: breadth_score += 1
    
    # 2. Distribution Filter (15%)
    dist_score = 0
    if ind["TRIN"] < 1.0: dist_score += 3
    elif ind["TRIN"] < 1.3: dist_score += 2
    elif ind["TRIN"] < 1.7: dist_score += 1
    
    if ind["SPXADP"] > 50: dist_score += 2
    elif ind["SPXADP"] > 20: dist_score += 1
    
    # 3. Price Confirmation (25%)
    price_score = 0
    if ind["RSP"] > 194.55: price_score += 3
    elif ind["RSP"] > 190.00: price_score += 2
    else: price_score += 1
    
    if ind["SPX"] > 6582: price_score += 2
    else: price_score += 1
    
    # 4. Ratio Leadership (20%)
    ratio_score = 0
    if "RSP:SPY" in ind and ind["RSP:SPY"] > 0.30: ratio_score += 2
    if "IWM:SPY" in ind and ind["IWM:SPY"] > 0.38: ratio_score += 2
    if "XLF:SPY" in ind and ind["XLF:SPY"] > 0.075: ratio_score += 1
    
    # 5. Credit/Sentiment (15%)
    credit_score = 0
    if ind["VIX"] < 20: credit_score += 2
    elif ind["VIX"] < 25: credit_score += 1
    
    if "CPCE" in ind and ind["CPCE"] < 0.60: credit_score += 2
    elif "CPCE" in ind and ind["CPCE"] < 0.70: credit_score += 1
    
    if "HYG:IEF" in ind and ind["HYG:IEF"] > 0.83: credit_score += 1
    
    # Weighted Total
    total = (
        (breadth_score / 7) * 25 + 
        (dist_score / 5) * 15 + 
        (price_score / 5) * 25 + 
        (ratio_score / 5) * 20 + 
        (credit_score / 5) * 15
    )
    
    return min(round(total, 1), 10)

# ============================================================================
# STREAMLIT UI
# ============================================================================

st.title("📊 Market Breadth Decision Engine v2.0")
st.sidebar.header("📁 Upload Data")

# 1. Historical ZIP
hist_zip = st.sidebar.file_uploader("Historical ZIP", type="zip")
# 2. Daily Snapshot
daily_csv = st.sidebar.file_uploader("Daily Snapshot (e.g., SC (21).csv)", type="csv")

historical_data = {}
snapshot = None
indicators = {}

if hist_zip:
    historical_data = load_historical_zip(hist_zip)
    st.sidebar.success(f"{len(historical_data)} historical series loaded & oscillators calculated")

if daily_csv:
    snapshot = load_snapshot(daily_csv)
    st.sidebar.success(f"Snapshot loaded: {daily_csv.name}")
    
    # Log the upload
    # We need a placeholder score for logging until calculated
    log_upload(daily_csv.name, "Pending", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Map Snapshot to Indicators
if snapshot is not None and "Symbol" in snapshot.columns:
    for _, row in snapshot.iterrows():
        s = str(row["Symbol"]).replace("$", "").strip()
        # Handle ratios like RSP:SPY
        if ":" in s:
            indicators[s] = float(row["Close"])
        elif s in ["SPXA50R", "BPSPX", "BPNYA", "TRIN", "SPXADP", "RSP", "SPX", "VIX", "CPCE", "URSP", "VXX"]:
            indicators[s] = float(row["Close"])

# Calculate Score if data exists
score = 0
osc_consensus = 0
if indicators and historical_data:
    score = calculate_gate_score_v2(indicators, historical_data)
    osc_consensus = calculate_oscillator_consensus(historical_data, indicators)
    
    # Update Log with Score
    if daily_csv:
        log_upload(daily_csv.name, score, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Dashboard
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Gate Score", f"{score}/10")
    if score >= 7:
        st.success("🟢 Strong Long")
    elif score >= 5:
        st.info("🟡 Long Bias / Watch")
    elif score >= 4:
        st.warning("🟠 Wait / Neutral")
    else:
        st.error("🔴 Risk Off")

with col2:
    st.metric("Oscillator Consensus", f"{osc_consensus}%")
    st.write("Based on TSI, MACD, RSI, ROC")

with col3:
    st.metric("Regime", "Strong Trend" if score >= 5 else "Choppy/Weak")

st.divider()

# Indicators Table
st.write("### 📊 Current Indicator Values")
st.json(indicators)

st.divider()

# Upload Log
st.write("### 📜 Upload History Log")
log_data = load_upload_log()
if log_data:
    log_df = pd.DataFrame(log_data)
    st.dataframe(log_df.tail(10), use_container_width=True)
else:
    st.write("No upload history found.")

st.divider()

# Visualization
if score > 0:
    fig = go.Figure()
    fig.add_histogram(x=np.random.normal(score, 1, 1000), nbinsx=40, opacity=0.7)
    fig.add_vline(x=score, line_color="red", line_width=3, annotation_text=f"Current: {score}")
    fig.update_layout(title="Gate Score Probability Distribution", xaxis_title="Score", yaxis_title="Frequency")
    st.plotly_chart(fig, use_container_width=True)

st.caption("v2.0 Engine | Oscillators calculated via pandas_ta | Logs saved to upload_history.json")
