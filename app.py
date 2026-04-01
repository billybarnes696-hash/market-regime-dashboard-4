
import io
import zipfile
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Market Breadth Decision Engine v4",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

REQUIRED_INDICATORS = [
    "SPXA50R", "BPSPX", "BPNYA", "TRIN", "SPXADP",
    "RSP", "SPX", "VIX", "CPCE", "RSP:SPY", "IWM:SPY",
]

OPTIONAL_INDICATORS = [
    "XLF:SPY", "HYG:IEF", "HYG:TLT", "URSP", "VXX", "SMH:SPY", "NYAD", "NYHL"
]

FILE_TO_KEY = {
    "SPXA50R": "SPXA50R",
    "BPSPX": "BPSPX",
    "BPNYA": "BPNYA",
    "TRIN": "TRIN",
    "SPXADP": "SPXADP",
    "RSP": "RSP",
    "SPX": "SPX",
    "VIX": "VIX",
    "CPCE": "CPCE",
    "RSP:SPY": "RSP:SPY",
    "IWM:SPY": "IWM:SPY",
    "XLF:SPY": "XLF:SPY",
    "HYG:IEF": "HYG:IEF",
    "HYG:TLT": "HYG:TLT",
    "URSP": "URSP",
    "VXX": "VXX",
    "SMH:SPY": "SMH:SPY",
    "NYAD": "NYAD",
    "NYHL": "NYHL",
}

BASE_DEFAULTS = {
    "SPXA50R": 29.8,
    "BPSPX": 38.0,
    "BPNYA": 44.4,
    "TRIN": 1.83,
    "SPXADP": 38.0,
    "RSP": 193.33,
    "SPX": 6582.0,
    "VIX": 23.55,
    "CPCE": 0.56,
    "RSP:SPY": 0.294,
    "IWM:SPY": 0.383,
    "XLF:SPY": 0.075,
    "HYG:IEF": 0.83,
    "HYG:TLT": 0.87,
    "URSP": 60.0,
    "VXX": 52.0,
    "SMH:SPY": 0.23,
    "NYAD": 0.0,
    "NYHL": 0.0,
}

DEFAULT_RSP_PIVOT = 194.55
DEFAULT_SPX_PIVOT = 6582.0

if "upload_history" not in st.session_state:
    st.session_state.upload_history = []
if "last_analysis_timestamp" not in st.session_state:
    st.session_state.last_analysis_timestamp = None
if "error_log" not in st.session_state:
    st.session_state.error_log = []

def normalize_symbol(s: str) -> str:
    return str(s).strip().replace("$", "").upper()

def safe_float(value, default=np.nan):
    try:
        return float(value)
    except Exception:
        return default

def record_error(context: str, exc: Exception) -> None:
    st.session_state.error_log.append({
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "context": context,
        "error": repr(exc),
        "traceback": traceback.format_exc(limit=3),
    })

def append_upload_history(filename: str, final_score: float, regime: str) -> None:
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": filename,
        "final_score": final_score,
        "regime": regime,
    }
    history = st.session_state.upload_history
    if not history or history[-1] != entry:
        history.append(entry)
    st.session_state.upload_history = history[-50:]

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def tsi(series: pd.Series, long_period: int = 25, short_period: int = 13, signal_period: int = 7):
    momentum = series.diff()
    smooth1 = ema(momentum, long_period)
    smooth2 = ema(smooth1, short_period)
    abs_momentum = momentum.abs()
    abs_smooth1 = ema(abs_momentum, long_period)
    abs_smooth2 = ema(abs_smooth1, short_period)
    tsi_line = 100 * (smooth2 / abs_smooth2.replace(0, np.nan))
    tsi_signal = ema(tsi_line, signal_period)
    return tsi_line, tsi_signal

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_gain = up.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = down.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def roc(series: pd.Series, period: int = 12) -> pd.Series:
    return ((series / series.shift(period)) - 1) * 100

def percent_b(series: pd.Series, period: int = 20, num_std: int = 2) -> pd.Series:
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return (series - lower) / (upper - lower).replace(0, np.nan)

def cci_proxy_from_close(series: pd.Series, period: int = 20) -> pd.Series:
    sma = series.rolling(period).mean()
    mad = (series - sma).abs().rolling(period).mean()
    return (series - sma) / (0.015 * mad.replace(0, np.nan))

def adx_proxy_from_close(series: pd.Series, period: int = 14):
    high = series * 1.02
    low = series * 0.98
    close = series
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = ema(tr, period).replace(0, np.nan)
    plus_di = 100 * ema(plus_dm, period) / atr
    minus_di = 100 * ema(minus_dm, period) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = ema(dx, period)
    return adx, plus_di, minus_di

@st.cache_data(show_spinner=False)
def compute_indicator_pack(series: pd.Series) -> pd.DataFrame:
    s = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    df = pd.DataFrame(index=s.index)
    df["Close"] = s
    tsi_line, tsi_signal = tsi(s)
    macd_line, macd_signal, macd_hist = macd(s)
    adx_proxy, plus_di, minus_di = adx_proxy_from_close(s)
    df["TSI"] = tsi_line
    df["TSI_SIGNAL"] = tsi_signal
    df["MACD"] = macd_line
    df["MACD_SIGNAL"] = macd_signal
    df["MACD_HIST"] = macd_hist
    df["RSI"] = rsi(s)
    df["ROC"] = roc(s)
    df["PCT_B"] = percent_b(s)
    df["CCI_PROXY"] = cci_proxy_from_close(s)
    df["ADX_PROXY"] = adx_proxy
    df["PLUS_DI_PROXY"] = plus_di
    df["MINUS_DI_PROXY"] = minus_di
    return df

@st.cache_data(show_spinner=False)
def load_historical_zip(zip_bytes: bytes):
    results = {}
    errors = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        csv_files = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        for member in csv_files:
            base = normalize_symbol(Path(member).stem)
            matched_key = next((candidate for candidate in FILE_TO_KEY if normalize_symbol(candidate) == base), None)
            if matched_key is None:
                continue
            try:
                with zf.open(member) as f:
                    df = pd.read_csv(f)
                date_col = next((c for c in ["Date", "date"] if c in df.columns), None)
                close_col = next((c for c in ["Close", "close", "Adj Close", "adj_close"] if c in df.columns), None)
                if date_col is None or close_col is None:
                    continue
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
                df = df.dropna(subset=[date_col, close_col]).sort_values(date_col).set_index(date_col)
                if len(df) < 30:
                    continue
                results[matched_key] = compute_indicator_pack(df[close_col])
            except Exception as e:
                errors.append(f"{member}: {e}")
    return results, errors

@st.cache_data(show_spinner=False)
def load_snapshot_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))

def build_snapshot_indicator_map(snapshot_df: pd.DataFrame) -> dict:
    indicators = dict(BASE_DEFAULTS)
    if "Symbol" not in snapshot_df.columns or "Close" not in snapshot_df.columns:
        return indicators
    lookup = {normalize_symbol(k): k for k in indicators}
    for _, row in snapshot_df.iterrows():
        symbol = normalize_symbol(row["Symbol"])
        if symbol in lookup:
            key = lookup[symbol]
            indicators[key] = safe_float(row["Close"], indicators[key])
    return indicators

def get_missing_required(indicators: dict) -> list:
    return [k for k in REQUIRED_INDICATORS if k not in indicators or pd.isna(indicators.get(k))]

def calculate_gate_score(ind: dict, rsp_pivot: float, spx_pivot: float):
    details = {}
    breadth = 0.0
    breadth += 1.0 if ind["SPXA50R"] > 40 else 0.7 if ind["SPXA50R"] > 30 else 0.4 if ind["SPXA50R"] > 20 else 0.1
    breadth += 0.8 if ind["BPSPX"] > 50 else 0.5 if ind["BPSPX"] > 40 else 0.2
    breadth += 0.7 if ind["BPNYA"] > 50 else 0.4 if ind["BPNYA"] > 40 else 0.2
    details["Breadth Momentum"] = min(breadth, 2.5)

    dist = 0.0
    dist += 1.0 if ind["TRIN"] < 1.0 else 0.7 if ind["TRIN"] < 1.3 else 0.3 if ind["TRIN"] < 1.7 else 0.0
    dist += 1.0 if ind["SPXADP"] > 50 else 0.7 if ind["SPXADP"] > 40 else 0.3 if ind["SPXADP"] > 20 else 0.0
    details["Distribution Filter"] = min(dist, 2.0)

    price = 0.0
    price += 1.0 if ind["RSP"] > rsp_pivot else 0.5 if ind["RSP"] > 190 else 0.2
    price += 1.0 if ind["SPX"] > spx_pivot else 0.5 if ind["SPX"] > spx_pivot * 0.99 else 0.2
    details["Price Confirmation"] = min(price, 2.0)

    ratio = 0.0
    ratio += 0.8 if ind["RSP:SPY"] > 0.300 else 0.5 if ind["RSP:SPY"] > 0.295 else 0.2
    ratio += 0.8 if ind["IWM:SPY"] > 0.390 else 0.5 if ind["IWM:SPY"] > 0.380 else 0.2
    ratio += 0.4 if ind.get("XLF:SPY", BASE_DEFAULTS["XLF:SPY"]) > 0.075 else 0.2
    details["Ratio Leadership"] = min(ratio, 2.0)

    sent = 0.0
    sent += 0.6 if ind["VIX"] < 20 else 0.4 if ind["VIX"] < 25 else 0.2
    sent += 0.5 if ind["CPCE"] < 0.60 else 0.3 if ind["CPCE"] < 0.70 else 0.1
    sent += 0.4 if ind.get("HYG:IEF", BASE_DEFAULTS["HYG:IEF"]) > 0.83 else 0.2
    details["Sentiment / Credit"] = min(sent, 1.5)

    gate_score = round(min(sum(details.values()), 10.0), 1)
    return gate_score, details

def calculate_oscillator_consensus(hist_data: dict):
    rows = []
    scoring_keys = ["SPXA50R", "BPSPX", "RSP", "TRIN", "VIX", "RSP:SPY", "IWM:SPY"]
    for key in scoring_keys:
        if key not in hist_data:
            continue
        df = hist_data[key]
        if df.empty:
            continue
        last = df.iloc[-1]
        points = 0
        max_points = 5
        if pd.notna(last["TSI"]) and pd.notna(last["TSI_SIGNAL"]) and last["TSI"] > last["TSI_SIGNAL"]:
            points += 1
        if pd.notna(last["MACD_HIST"]) and last["MACD_HIST"] > 0:
            points += 1
        if pd.notna(last["RSI"]) and last["RSI"] > 50:
            points += 1
        if pd.notna(last["ROC"]) and last["ROC"] > 0:
            points += 1
        if pd.notna(last["PCT_B"]) and last["PCT_B"] > 0.5:
            points += 1
        if key in ["TRIN", "VIX"]:
            points = max_points - points
        rows.append({"Indicator": key, "Bullish Points": points, "Max": max_points})
    if not rows:
        return 0.0, pd.DataFrame()
    detail = pd.DataFrame(rows)
    consensus_pct = round(100 * detail["Bullish Points"].sum() / detail["Max"].sum(), 1)
    return consensus_pct, detail

def oscillator_pct_to_score(consensus_pct: float) -> float:
    return round((consensus_pct / 100.0) * 10.0, 1)

def calculate_final_score(gate_score: float, oscillator_score: float, gate_weight: float = 0.7, osc_weight: float = 0.3) -> float:
    return round((gate_score * gate_weight) + (oscillator_score * osc_weight), 1)

def classify_regime(score: float) -> str:
    if score >= 7.0:
        return "Strong Long"
    if score >= 5.5:
        return "Long Bias / Confirmation"
    if score >= 4.0:
        return "Wait / Neutral"
    return "Risk Off"

def suggest_position_size(score: float) -> str:
    if score >= 7.5:
        return "75-100%"
    if score >= 6.0:
        return "40-60%"
    if score >= 4.5:
        return "10-25%"
    return "0-10%"

def detect_repair_signal(ind: dict, gate_score: float, final_score: float) -> str:
    if ind["SPXA50R"] > 30 and ind["BPSPX"] > 40 and gate_score >= 5.5 and final_score >= 6.0:
        return "Confirmed Repair"
    if ind["SPXA50R"] > 20 and ind["BPSPX"] > 30 and gate_score >= 4.5:
        return "Early Repair"
    if gate_score < 4.0:
        return "No Repair"
    return "Attempting Repair"

def calc_proxy_nymo(ind: dict) -> float:
    nyad = safe_float(ind.get("NYAD", 0.0), 0.0)
    spxadp = safe_float(ind.get("SPXADP", 0.0), 0.0)
    raw = (0.6 * nyad / 1000.0) + (0.4 * (spxadp - 50) / 10.0)
    return round(100 * np.tanh(raw / 5.0), 1)

@st.cache_data(show_spinner=False)
def build_real_historical_score_series(hist_data: dict, rsp_pivot: float, spx_pivot: float):
    needed = ["SPXA50R", "BPSPX", "BPNYA", "TRIN", "SPXADP", "RSP", "SPX", "VIX", "CPCE", "RSP:SPY", "IWM:SPY"]
    available = [k for k in needed if k in hist_data]
    if len(available) < 8:
        return pd.DataFrame()

    union_index = None
    for key in available:
        union_index = hist_data[key].index if union_index is None else union_index.union(hist_data[key].index)

    close_df = pd.DataFrame(index=union_index)
    extra_keys = [k for k in ["XLF:SPY", "HYG:IEF"] if k in hist_data]
    for key in available + extra_keys:
        close_df[key] = hist_data[key]["Close"]

    close_df = close_df.sort_index().ffill().dropna(subset=available)
    if close_df.empty:
        return pd.DataFrame()

    rows = []
    for dt, row in close_df.iterrows():
        snapshot = dict(BASE_DEFAULTS)
        for key in close_df.columns:
            if pd.notna(row[key]):
                snapshot[key] = float(row[key])

        gate_score, _ = calculate_gate_score(snapshot, rsp_pivot, spx_pivot)

        osc_points = 0
        osc_max = 0
        osc_keys = [k for k in ["SPXA50R", "BPSPX", "RSP", "TRIN", "VIX", "RSP:SPY", "IWM:SPY"] if k in hist_data]
        for key in osc_keys:
            hist_slice = hist_data[key].loc[:dt]
            if hist_slice.empty:
                continue
            last = hist_slice.iloc[-1]
            pts = 0
            if pd.notna(last["TSI"]) and pd.notna(last["TSI_SIGNAL"]) and last["TSI"] > last["TSI_SIGNAL"]:
                pts += 1
            if pd.notna(last["MACD_HIST"]) and last["MACD_HIST"] > 0:
                pts += 1
            if pd.notna(last["RSI"]) and last["RSI"] > 50:
                pts += 1
            if pd.notna(last["ROC"]) and last["ROC"] > 0:
                pts += 1
            if pd.notna(last["PCT_B"]) and last["PCT_B"] > 0.5:
                pts += 1
            if key in ["TRIN", "VIX"]:
                pts = 5 - pts
            osc_points += pts
            osc_max += 5

        consensus_pct = 0.0 if osc_max == 0 else (100.0 * osc_points / osc_max)
        oscillator_score = oscillator_pct_to_score(consensus_pct)
        final_score = calculate_final_score(gate_score, oscillator_score)

        rows.append({
            "Date": dt,
            "GateScore": gate_score,
            "OscillatorConsensusPct": round(consensus_pct, 1),
            "OscillatorScore": oscillator_score,
            "FinalScore": final_score,
        })

    return pd.DataFrame(rows).set_index("Date").sort_index()

st.title("📊 Market Breadth Decision Engine v4")
st.caption("V4 combines gate score and oscillator score into a single final score.")

with st.sidebar:
    st.header("📁 Upload Data")
    hist_zip = st.file_uploader("1) Historical ZIP", type=["zip"])
    daily_csv = st.file_uploader("2) Daily Snapshot CSV", type=["csv"])
    st.markdown("---")
    st.subheader("Weights")
    gate_weight = st.slider("Gate Score Weight", 0.0, 1.0, 0.7, 0.05)
    osc_weight = round(1.0 - gate_weight, 2)
    st.caption(f"Oscillator Weight auto-set to {osc_weight}")
    st.markdown("---")
    st.subheader("Manual Overrides")
    rsp_pivot = st.number_input("RSP Pivot", value=float(DEFAULT_RSP_PIVOT), step=0.01)
    spx_pivot = st.number_input("SPX Pivot", value=float(DEFAULT_SPX_PIVOT), step=1.0)
    show_debug = st.checkbox("Show debug / error log", value=False)

historical_data = {}
historical_errors = []
snapshot_df = None
indicators = dict(BASE_DEFAULTS)

if hist_zip is not None:
    try:
        historical_data, historical_errors = load_historical_zip(hist_zip.getvalue())
        st.sidebar.success(f"Loaded {len(historical_data)} historical series")
    except Exception as e:
        record_error("load_historical_zip", e)
        st.sidebar.error("Could not read historical ZIP.")

if daily_csv is not None:
    try:
        snapshot_df = load_snapshot_csv(daily_csv.getvalue())
        indicators = build_snapshot_indicator_map(snapshot_df)
        st.sidebar.success(f"Loaded snapshot: {daily_csv.name}")
    except Exception as e:
        record_error("load_snapshot_csv", e)
        st.sidebar.error("Could not read daily snapshot CSV.")

missing_required = get_missing_required(indicators)
gate_score, score_details = calculate_gate_score(indicators, rsp_pivot, spx_pivot)
osc_consensus_pct, osc_detail = calculate_oscillator_consensus(historical_data)
oscillator_score = oscillator_pct_to_score(osc_consensus_pct)
final_score = calculate_final_score(gate_score, oscillator_score, gate_weight, osc_weight)
regime = classify_regime(final_score)
position_size = suggest_position_size(final_score)
repair_state = detect_repair_signal(indicators, gate_score, final_score)
proxy_nymo = calc_proxy_nymo(indicators)

historical_score_series = pd.DataFrame()
if historical_data:
    try:
        historical_score_series = build_real_historical_score_series(historical_data, rsp_pivot, spx_pivot)
    except Exception as e:
        record_error("build_real_historical_score_series", e)

current_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
prior_ts = st.session_state.last_analysis_timestamp
st.session_state.last_analysis_timestamp = current_ts

if daily_csv is not None:
    append_upload_history(daily_csv.name, final_score, regime)

if historical_errors:
    with st.expander("ZIP parse warnings"):
        for err in historical_errors[:25]:
            st.write(f"• {err}")

if missing_required:
    st.warning("Missing required indicators from snapshot: " + ", ".join(missing_required))

if historical_data and historical_score_series.empty:
    st.warning("ZIP loaded, but not enough overlapping history was available to build a real historical score series.")
elif not historical_data:
    st.warning("Upload a valid historical ZIP to activate oscillator context and real historical score distribution.")

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Gate Score", f"{gate_score}/10")
with c2:
    st.metric("Oscillator Score", f"{oscillator_score}/10", delta=f"{osc_consensus_pct}% bullish")
with c3:
    st.metric("Final Score", f"{final_score}/10")
    if final_score >= 7:
        st.success("🟢 Strong Long")
    elif final_score >= 5.5:
        st.info("🔵 Long Bias / Confirmation")
    elif final_score >= 4:
        st.warning("🟡 Wait / Neutral")
    else:
        st.error("🔴 Risk Off")
with c4:
    st.metric("Repair State", repair_state)
    st.caption(f"Target size: {position_size}")
with c5:
    st.metric("Proxy NYMO", proxy_nymo)
    st.caption(f"Prior analysis: {prior_ts or 'N/A'}")

tab1, tab2, tab3, tab4 = st.tabs([
    "Summary",
    "Historical Score Engine",
    "Indicators & Oscillators",
    "Logs / Debug",
])

with tab1:
    left, right = st.columns([1, 1.2])
    with left:
        st.subheader("Score Breakdown")
        score_df = pd.DataFrame({
            "Component": list(score_details.keys()),
            "Gate Score Points": list(score_details.values())
        })
        st.dataframe(score_df, use_container_width=True, hide_index=True)

        st.subheader("Current Decision")
        if final_score >= 7.0:
            st.success(f'''
**Position**: {position_size} long  
**Bias**: Confirmation already present  
**Repair State**: {repair_state}  
**Vehicles**: RSP / URSP  
**Stops**: Use pivot-based stops, tighten if TRIN rises or ratios weaken
''')
        elif final_score >= 5.5:
            st.info(f'''
**Position**: {position_size} long  
**Bias**: Add only on confirmation  
**Repair State**: {repair_state}  
**Vehicles**: RSP / URSP small-to-medium size  
**Stops**: Tighten risk if breadth fades
''')
        elif final_score >= 4.0:
            st.warning(f'''
**Position**: {position_size} probe or cash  
**Bias**: Wait for cleaner confirmation  
**Repair State**: {repair_state}  
**Stops**: Small size only, tighter risk
''')
        else:
            st.error(f'''
**Position**: {position_size} exposure  
**Bias**: Avoid forcing longs  
**Repair State**: {repair_state}  
**Focus**: Preserve capital until structure improves
''')

    with right:
        st.subheader("Current Score Gauge")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=final_score,
            number={"suffix": "/10"},
            gauge={
                "axis": {"range": [0, 10]},
                "bar": {"thickness": 0.25},
                "steps": [
                    {"range": [0, 4], "color": "#f8d7da"},
                    {"range": [4, 5.5], "color": "#fff3cd"},
                    {"range": [5.5, 7], "color": "#d1ecf1"},
                    {"range": [7, 10], "color": "#d4edda"},
                ],
            },
            title={"text": "Final Score"},
        ))
        fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.subheader("Current Indicator Values")
        indicator_rows = []
        for key in REQUIRED_INDICATORS + OPTIONAL_INDICATORS:
            if key in indicators:
                indicator_rows.append({"Indicator": key, "Current": indicators[key]})
        st.dataframe(pd.DataFrame(indicator_rows), use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Historical Score Engine")
    if historical_score_series.empty:
        st.write("No real historical score series available yet. Upload a richer ZIP with overlapping history.")
    else:
        latest_final = float(historical_score_series["FinalScore"].iloc[-1])
        percentile = float((historical_score_series["FinalScore"] <= latest_final).mean() * 100)
        hc1, hc2, hc3 = st.columns(3)
        with hc1:
            st.metric("Historical Percentile", f"{percentile:.1f}th")
        with hc2:
            st.metric("Historical Mean", f"{historical_score_series['FinalScore'].mean():.2f}")
        with hc3:
            st.metric("Historical Std Dev", f"{historical_score_series['FinalScore'].std():.2f}")

        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=historical_score_series.index,
            y=historical_score_series["FinalScore"],
            mode="lines",
            name="Final Score",
        ))
        fig_ts.add_hline(y=7.0, line_dash="dash", line_color="green")
        fig_ts.add_hline(y=5.5, line_dash="dash", line_color="blue")
        fig_ts.add_hline(y=4.0, line_dash="dash", line_color="orange")
        fig_ts.update_layout(height=380, xaxis_title="Date", yaxis_title="Final Score")
        st.plotly_chart(fig_ts, use_container_width=True)

        fig_hist = go.Figure()
        fig_hist.add_histogram(x=historical_score_series["FinalScore"], nbinsx=35, opacity=0.8)
        fig_hist.add_vline(x=final_score, line_color="red", line_width=3, annotation_text=f"Current: {final_score}")
        fig_hist.update_layout(height=380, xaxis_title="Final Score", yaxis_title="Frequency")
        st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("Latest Historical Scores")
        st.dataframe(
            historical_score_series.tail(15).reset_index().rename(columns={"index": "Date"}),
            use_container_width=True,
            hide_index=True,
        )

with tab3:
    left, right = st.columns(2)
    with left:
        st.subheader("Oscillator Consensus Detail")
        if osc_detail.empty:
            st.write("Upload historical ZIP to compute oscillator consensus.")
        else:
            st.dataframe(osc_detail, use_container_width=True, hide_index=True)

        st.subheader("Composite Math")
        comp_df = pd.DataFrame([
            {"Metric": "Gate Score", "Value": gate_score},
            {"Metric": "Oscillator Consensus %", "Value": osc_consensus_pct},
            {"Metric": "Oscillator Score", "Value": oscillator_score},
            {"Metric": "Gate Weight", "Value": gate_weight},
            {"Metric": "Oscillator Weight", "Value": osc_weight},
            {"Metric": "Final Score", "Value": final_score},
        ])
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    with right:
        st.subheader("Historical Latest Indicator Snapshot")
        if not historical_data:
            st.write("Upload historical ZIP to see historical oscillator values.")
        else:
            hist_rows = []
            for key, df in historical_data.items():
                last = df.iloc[-1]
                hist_rows.append({
                    "Indicator": key,
                    "Last Date": df.index[-1].strftime("%Y-%m-%d"),
                    "Close": round(float(last["Close"]), 4),
                    "TSI": round(float(last["TSI"]), 2) if pd.notna(last["TSI"]) else np.nan,
                    "MACD Hist": round(float(last["MACD_HIST"]), 2) if pd.notna(last["MACD_HIST"]) else np.nan,
                    "RSI": round(float(last["RSI"]), 2) if pd.notna(last["RSI"]) else np.nan,
                    "ROC": round(float(last["ROC"]), 2) if pd.notna(last["ROC"]) else np.nan,
                    "%B": round(float(last["PCT_B"]), 2) if pd.notna(last["PCT_B"]) else np.nan,
                    "CCI Proxy": round(float(last["CCI_PROXY"]), 2) if pd.notna(last["CCI_PROXY"]) else np.nan,
                    "ADX Proxy": round(float(last["ADX_PROXY"]), 2) if pd.notna(last["ADX_PROXY"]) else np.nan,
                })
            st.dataframe(pd.DataFrame(hist_rows).sort_values("Indicator"), use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Upload History")
    if st.session_state.upload_history:
        st.dataframe(pd.DataFrame(st.session_state.upload_history[-10:]).iloc[::-1], use_container_width=True, hide_index=True)
    else:
        st.write("No upload history yet.")

    if show_debug:
        st.subheader("Debug / Error Log")
        if st.session_state.error_log:
            st.dataframe(pd.DataFrame(st.session_state.error_log), use_container_width=True, hide_index=True)
        else:
            st.write("No errors logged.")

st.caption("ADX is labeled as ADX Proxy because this build uses close-derived synthetic ranges when OHLC is unavailable.")
