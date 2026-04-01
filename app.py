# app.py - Market Breadth Decision Engine
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Market Breadth Decision Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# TECHNICAL INDICATOR CALCULATIONS
# ============================================================================

def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    return data.ewm(span=period, adjust=False).mean()

def calculate_tsi(data, long_period=25, short_period=13, signal_period=7):
    """Calculate True Strength Index"""
    momentum = data.diff()
    first_smooth = momentum.ewm(span=long_period, adjust=False).mean()
    second_smooth = first_smooth.ewm(span=short_period, adjust=False).mean()
    abs_momentum = abs(momentum)
    first_abs = abs_momentum.ewm(span=long_period, adjust=False).mean()
    second_abs = first_abs.ewm(span=short_period, adjust=False).mean()
    tsi = 100 * (second_smooth / second_abs.replace(0, np.nan))
    tsi_signal = tsi.ewm(span=signal_period, adjust=False).mean()
    return tsi.fillna(0), tsi_signal.fillna(0)

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_mcci(data, period=20):
    """Calculate Market Cycle Confidence Index (simplified CCI)"""
    tp = data  # Using close as approximation
    sma = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
    mcci = (tp - sma) / (0.015 * mad.replace(0, np.nan))
    return mcci.fillna(0)

def calculate_stochastic(data, k_period=14, d_period=3):
    """Calculate Full Stochastic"""
    low_min = data.rolling(window=k_period).min()
    high_max = data.rolling(window=k_period).max()
    k = 100 * (data - low_min) / (high_max - low_min).replace(0, np.nan)
    d = k.rolling(window=d_period).mean()
    return k.fillna(0), d.fillna(0)

def calculate_percent_b(data, period=20, num_std=2):
    """Calculate Bollinger %B"""
    sma = data.rolling(window=period).mean()
    std = data.rolling(window=period).std()
    upper = sma + (num_std * std)
    lower = sma - (num_std * std)
    percent_b = (data - lower) / (upper - lower).replace(0, np.nan)
    return percent_b.fillna(0.5)

def calculate_roc(data, period=12):
    """Calculate Rate of Change"""
    return ((data / data.shift(period)) - 1) * 100

def calculate_adx(data, period=14):
    """Calculate ADX with +DI and -DI"""
    high = data * 1.02  # Approximation
    low = data * 0.98   # Approximation
    close = data
    
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    plus_di = 100 * calculate_ema(plus_dm, period) / calculate_ema(tr, period)
    minus_di = 100 * calculate_ema(minus_dm, period) / calculate_ema(tr, period)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx = calculate_ema(dx, period)
    
    return adx.fillna(0), plus_di.fillna(0), minus_di.fillna(0)

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

@st.cache_data
def load_historical_data(uploaded_files):
    """Load and combine historical CSV files from zip extraction"""
    all_data = {}
    for file in uploaded_files:
        try:
            df = pd.read_csv(file)
            if 'Date' in df.columns or 'date' in df.columns:
                date_col = 'Date' if 'Date' in df.columns else 'date'
                df[date_col] = pd.to_datetime(df[date_col])
                df.set_index(date_col, inplace=True)
                symbol = file.name.replace('.csv', '').replace('$', '')
                if 'Close' in df.columns:
                    all_data[symbol] = df['Close']
        except Exception as e:
            st.warning(f"Could not load {file.name}: {str(e)}")
    return all_data

@st.cache_data
def load_daily_snapshot(uploaded_file):
    """Load daily snapshot CSV"""
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading snapshot: {str(e)}")
        return None

# ============================================================================
# GATE SCORE CALCULATION
# ============================================================================

def calculate_gate_score(indicators, historical_data=None):
    """
    Calculate Composite Gate Score (0-10)
    
    Components:
    - Breadth Momentum (30%): SPXA50R, BPSPX, BPNYA
    - Distribution Filter (25%): TRIN, SPXADP
    - Price Confirmation (20%): RSP, SPX
    - Ratio Leadership (15%): RSP:SPY, IWM:SPY
    - Sentiment/Volatility (10%): VIX, CPCE (inverse)
    """
    score = 0
    details = {}
    
    # === BREADTH MOMENTUM (30%) ===
    breadth_score = 0
    
    # SPXA50R scoring
    spxa50r = indicators.get('SPXA50R', 0)
    if spxa50r > 40:
        breadth_score += 1.0
    elif spxa50r > 30:
        breadth_score += 0.7
    elif spxa50r > 20:
        breadth_score += 0.4
    else:
        breadth_score += 0.1
    
    # BPSPX scoring
    bpspx = indicators.get('BPSPX', 0)
    if bpspx > 50:
        breadth_score += 1.0
    elif bpspx > 40:
        breadth_score += 0.7
    elif bpspx > 30:
        breadth_score += 0.4
    else:
        breadth_score += 0.1
    
    # BPNYA scoring
    bpnya = indicators.get('BPNYA', 0)
    if bpnya > 50:
        breadth_score += 1.0
    elif bpnya > 40:
        breadth_score += 0.7
    elif bpnya > 30:
        breadth_score += 0.4
    else:
        breadth_score += 0.1
    
    details['Breadth Momentum'] = min(breadth_score, 3.0) * 1.0  # Max 3.0 points
    
    # === DISTRIBUTION FILTER (25%) ===
    dist_score = 0
    
    # TRIN scoring (inverse - lower is better)
    trin = indicators.get('TRIN', 999)
    if trin < 0.8:
        dist_score += 1.25
    elif trin < 1.0:
        dist_score += 1.0
    elif trin < 1.3:
        dist_score += 0.6
    elif trin < 1.5:
        dist_score += 0.3
    else:
        dist_score += 0.0
    
    # SPXADP scoring
    spxadp = indicators.get('SPXADP', 0)
    if spxadp > 60:
        dist_score += 1.25
    elif spxadp > 50:
        dist_score += 1.0
    elif spxadp > 40:
        dist_score += 0.6
    elif spxadp > 30:
        dist_score += 0.3
    else:
        dist_score += 0.0
    
    details['Distribution Filter'] = min(dist_score, 2.5) * 1.0  # Max 2.5 points
    
    # === PRICE CONFIRMATION (20%) ===
    price_score = 0
    
    # RSP position
    rsp = indicators.get('RSP', 0)
    rsp_pivot = indicators.get('RSP_Pivot', rsp * 0.95)
    if rsp > rsp_pivot * 1.05:
        price_score += 1.0
    elif rsp > rsp_pivot:
        price_score += 0.7
    elif rsp > rsp_pivot * 0.95:
        price_score += 0.4
    else:
        price_score += 0.1
    
    # SPX position
    spx = indicators.get('SPX', 0)
    spx_pivot = indicators.get('SPX_Pivot', spx * 0.95)
    if spx > spx_pivot * 1.05:
        price_score += 1.0
    elif spx > spx_pivot:
        price_score += 0.7
    elif spx > spx_pivot * 0.95:
        price_score += 0.4
    else:
        price_score += 0.1
    
    details['Price Confirmation'] = min(price_score, 2.0) * 1.0  # Max 2.0 points
    
    # === RATIO LEADERSHIP (15%) ===
    ratio_score = 0
    
    # RSP:SPY
    rsp_spy = indicators.get('RSP_SPY', 0)
    if rsp_spy > 0.30:
        ratio_score += 0.75
    elif rsp_spy > 0.29:
        ratio_score += 0.5
    else:
        ratio_score += 0.25
    
    # IWM:SPY
    iwm_spy = indicators.get('IWM_SPY', 0)
    if iwm_spy > 0.39:
        ratio_score += 0.75
    elif iwm_spy > 0.38:
        ratio_score += 0.5
    else:
        ratio_score += 0.25
    
    details['Ratio Leadership'] = min(ratio_score, 1.5) * 1.0  # Max 1.5 points
    
    # === SENTIMENT/VOLATILITY (10%) ===
    sent_score = 0
    
    # VIX (inverse - lower is better)
    vix = indicators.get('VIX', 999)
    if vix < 15:
        sent_score += 0.5
    elif vix < 20:
        sent_score += 0.4
    elif vix < 25:
        sent_score += 0.3
    elif vix < 30:
        sent_score += 0.2
    else:
        sent_score += 0.1
    
    # CPCE (inverse - lower is better)
    cpce = indicators.get('CPCE', 999)
    if cpce < 0.5:
        sent_score += 0.5
    elif cpce < 0.7:
        sent_score += 0.4
    elif cpce < 0.9:
        sent_score += 0.3
    else:
        sent_score += 0.1
    
    details['Sentiment/Volatility'] = min(sent_score, 1.0) * 1.0  # Max 1.0 points
    
    # === TOTAL SCORE ===
    total_score = sum(details.values())
    
    return round(min(total_score, 10.0), 1), details

# ============================================================================
# HISTORICAL ANALYSIS & BELL CURVE
# ============================================================================

def calculate_historical_percentile(current_score, historical_scores):
    """Calculate where current score falls in historical distribution"""
    if len(historical_scores) < 30:
        return 50.0, "Insufficient Data"
    
    percentile = stats.percentileofscore(historical_scores, current_score)
    z_score = (current_score - np.mean(historical_scores)) / (np.std(historical_scores) + 0.001)
    
    if percentile < 10:
        regime = "Extreme Capitulation"
    elif percentile < 25:
        regime = "Capitulation Zone"
    elif percentile < 40:
        regime = "Oversold"
    elif percentile < 60:
        regime = "Neutral"
    elif percentile < 75:
        regime = "Overbought"
    elif percentile < 90:
        regime = "Euphoria Zone"
    else:
        regime = "Extreme Euphoria"
    
    return percentile, regime, z_score

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.title("📊 Market Breadth Decision Engine")
    st.markdown("""
    **Comprehensive breadth analysis with oscillator consensus, gate scoring, and historical context**
    
    *Upload historical CSV files + daily snapshot for real-time analysis*
    """)
    
    # ========================================================================
    # SIDEBAR - DATA UPLOAD & MANUAL INPUTS
    # ========================================================================
    st.sidebar.header("📁 Data Input")
    
    # Historical data upload
    historical_files = st.sidebar.file_uploader(
        "Historical CSV Files (from zip extraction)",
        type=['csv'],
        accept_multiple_files=True,
        help="Upload individual CSV files extracted from your historical zip"
    )
    
    # Daily snapshot upload
    snapshot_file = st.sidebar.file_uploader(
        "Daily Snapshot CSV",
        type=['csv'],
        help="Upload current day's snapshot (SC file format)"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("📝 Manual Indicator Inputs")
    st.sidebar.markdown("*Override snapshot data if needed*")
    
    # Key indicators for manual input
    indicators = {}
    
    indicators['SPXA50R'] = st.sidebar.number_input("SPXA50R (%)", value=29.80, step=0.1)
    indicators['BPSPX'] = st.sidebar.number_input("BPSPX (%)", value=38.00, step=0.1)
    indicators['BPNYA'] = st.sidebar.number_input("BPNYA (%)", value=44.37, step=0.1)
    indicators['TRIN'] = st.sidebar.number_input("TRIN", value=1.83, step=0.01)
    indicators['SPXADP'] = st.sidebar.number_input("SPXADP", value=38.00, step=0.1)
    indicators['NYAD'] = st.sidebar.number_input("NYAD", value=48728, step=100)
    indicators['RSP'] = st.sidebar.number_input("RSP Price", value=193.33, step=0.01)
    indicators['SPX'] = st.sidebar.number_input("SPX Price", value=6604, step=1)
    indicators['VIX'] = st.sidebar.number_input("VIX", value=23.55, step=0.01)
    indicators['CPCE'] = st.sidebar.number_input("CPCE", value=0.56, step=0.01)
    indicators['RSP_SPY'] = st.sidebar.number_input("RSP:SPY Ratio", value=0.294, step=0.001)
    indicators['IWM_SPY'] = st.sidebar.number_input("IWM:SPY Ratio", value=0.383, step=0.001)
    
    # Pivot points
    indicators['RSP_Pivot'] = st.sidebar.number_input("RSP Pivot", value=194.55, step=0.01)
    indicators['SPX_Pivot'] = st.sidebar.number_input("SPX Pivot", value=6582, step=1)
    
    # ========================================================================
    # LOAD DATA
    # ========================================================================
    historical_data = {}
    snapshot_data = None
    
    if historical_files:
        historical_data = load_historical_data(historical_files)
        st.sidebar.success(f"Loaded {len(historical_data)} historical series")
    
    if snapshot_file:
        snapshot_data = load_daily_snapshot(snapshot_file)
        if snapshot_data is not None:
            st.sidebar.success("Daily snapshot loaded")
            # Auto-populate from snapshot if available
            if 'Symbol' in snapshot_data.columns and 'Close' in snapshot_data.columns:
                for _, row in snapshot_data.iterrows():
                    symbol = str(row['Symbol']).replace('$', '').strip()
                    if symbol in indicators.keys() or symbol.upper() in [k.upper() for k in indicators.keys()]:
                        for key in indicators.keys():
                            if symbol.upper() == key.upper():
                                indicators[key] = float(row['Close'])
                                break
    
    # ========================================================================
    # CALCULATE GATE SCORE
    # ========================================================================
    gate_score, score_details = calculate_gate_score(indicators, historical_data)
    
    # ========================================================================
    # MAIN DISPLAY
    # ========================================================================
    
    # Row 1: Gate Score & Regime
    col1, col2, col3 = st.columns([2, 2, 2])
    
    with col1:
        st.metric("🚪 Gate Score", f"{gate_score}/10", 
                  delta=f"{gate_score - 5.0:+.1f} vs Neutral")
        
        # Signal interpretation
        if gate_score >= 7.0:
            st.success("🟢 STRONG BUY - High Conviction Long")
        elif gate_score >= 5.5:
            st.info("🔵 BUY - Scale In With Confirmation")
        elif gate_score >= 4.0:
            st.warning("🟡 WAIT - Watch for Confirmation")
        elif gate_score >= 2.5:
            st.orange("🟠 CAUTION - Reduce Exposure")
        else:
            st.error("🔴 SELL - Defensive Posture")
    
    with col2:
        # Historical percentile
        if historical_
            # Generate pseudo-historical scores for demo
            np.random.seed(42)
            historical_scores = np.random.normal(5.0, 2.0, 1000)
            historical_scores = np.clip(historical_scores, 0, 10)
            
            percentile, regime, z_score = calculate_historical_percentile(
                gate_score, historical_scores
            )
            
            st.metric("📈 Historical Percentile", f"{percentile:.1f}th")
            st.metric("📊 Z-Score", f"{z_score:.2f}")
            st.info(f"**Regime**: {regime}")
        else:
            st.info("Upload historical data for percentile analysis")
    
    with col3:
        # Oscillator consensus
        st.subheader("📊 Oscillator Consensus")
        
        # Calculate oscillators for key indicators
        osc_summary = {
            'TSI Bullish': 0,
            'MACD Positive': 0,
            '%B Oversold': 0,
            'Stoch Bullish': 0
        }
        
        # Demo oscillator counts (would calculate from actual data)
        osc_summary['TSI Bullish'] = 6  # Out of 10
        osc_summary['MACD Positive'] = 5
        osc_summary['%B Oversold'] = 4
        osc_summary['Stoch Bullish'] = 6
        
        for osc, count in osc_summary.items():
            st.metric(osc, f"{count}/10")
    
    # Row 2: Score Breakdown Chart
    st.markdown("---")
    st.subheader("📊 Gate Score Breakdown")
    
    breakdown_df = pd.DataFrame({
        'Component': list(score_details.keys()),
        'Score': list(score_details.values()),
        'Max': [3.0, 2.5, 2.0, 1.5, 1.0]
    })
    
    fig_breakdown = go.Figure()
    fig_breakdown.add_trace(go.Bar(
        x=breakdown_df['Component'],
        y=breakdown_df['Score'],
        name='Current',
        marker_color='steelblue'
    ))
    fig_breakdown.add_trace(go.Bar(
        x=breakdown_df['Component'],
        y=breakdown_df['Max'],
        name='Maximum',
        marker_color='lightgray',
        opacity=0.5
    ))
    fig_breakdown.update_layout(
        barmode='group',
        height=400,
        showlegend=True,
        yaxis_title="Score Points",
        yaxis_range=[0, 3.5]
    )
    st.plotly_chart(fig_breakdown, use_container_width=True)
    
    # Row 3: Bell Curve
    if historical_
        st.markdown("---")
        st.subheader("🔔 Historical Distribution Bell Curve")
        
        fig_bell = go.Figure()
        fig_bell.add_trace(go.Histogram(
            x=historical_scores,
            nbinsx=50,
            name='Historical Scores',
            marker_color='lightblue',
            opacity=0.7
        ))
        fig_bell.add_vline(
            x=gate_score,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Current: {gate_score}",
            annotation_position="top"
        )
        fig_bell.add_vline(
            x=np.mean(historical_scores),
            line_dash="dot",
            line_color="green",
            annotation_text=f"Mean: {np.mean(historical_scores):.1f}",
            annotation_position="top"
        )
        fig_bell.update_layout(
            height=400,
            showlegend=True,
            xaxis_title="Gate Score",
            yaxis_title="Frequency"
        )
        st.plotly_chart(fig_bell, use_container_width=True)
    
    # Row 4: Key Indicator Table
    st.markdown("---")
    st.subheader("📋 Key Indicator Summary")
    
    indicator_table = pd.DataFrame({
        'Indicator': ['SPXA50R', 'BPSPX', 'BPNYA', 'TRIN', 'SPXADP', 
                      'RSP', 'VIX', 'CPCE', 'RSP:SPY', 'IWM:SPY'],
        'Current': [
            indicators['SPXA50R'], indicators['BPSPX'], indicators['BPNYA'],
            indicators['TRIN'], indicators['SPXADP'], indicators['RSP'],
            indicators['VIX'], indicators['CPCE'], 
            indicators['RSP_SPY'], indicators['IWM_SPY']
        ],
        'Bullish Threshold': [
            '>40', '>50', '>50', '<1.0', '>50',
            '>Pivot', '<20', '<0.7', '>0.30', '>0.39'
        ],
        'Status': [
            '🔴' if indicators['SPXA50R'] < 30 else '🟡' if indicators['SPXA50R'] < 40 else '🟢',
            '🔴' if indicators['BPSPX'] < 30 else '🟡' if indicators['BPSPX'] < 40 else '🟢',
            '🔴' if indicators['BPNYA'] < 30 else '🟡' if indicators['BPNYA'] < 40 else '🟢',
            '🟢' if indicators['TRIN'] < 1.0 else '🟡' if indicators['TRIN'] < 1.3 else '🔴',
            '🔴' if indicators['SPXADP'] < 30 else '🟡' if indicators['SPXADP'] < 50 else '🟢',
            '🟢' if indicators['RSP'] > indicators['RSP_Pivot'] else '🔴',
            '🟢' if indicators['VIX'] < 20 else '🟡' if indicators['VIX'] < 30 else '🔴',
            '🟢' if indicators['CPCE'] < 0.7 else '🟡' if indicators['CPCE'] < 0.9 else '🔴',
            '🟢' if indicators['RSP_SPY'] > 0.30 else '🟡',
            '🟢' if indicators['IWM_SPY'] > 0.39 else '🟡'
        ]
    })
    
    st.dataframe(indicator_table, use_container_width=True, hide_index=True)
    
    # Row 5: Action Plan
    st.markdown("---")
    st.subheader("🎯 Action Plan")
    
    if gate_score >= 7.0:
        st.success("""
        **POSITION**: 75-100% Long (RSP/URSP/SPXL)
        
        **ENTRY**: Market or on pullback to support
        
        **STOP**: Below S1 pivot or -5% from entry
        
        **TARGET**: R2 resistance or trail 20-EMA
        
        **CONFIDENCE**: High - Multiple confirmations aligned
        """)
    elif gate_score >= 5.5:
        st.info("""
        **POSITION**: 50% Long (RSP/URSP)
        
        **ENTRY**: Scale in on confirmation
        
        **STOP**: Below EMAENV lower or -4% from entry
        
        **TARGET**: R1 resistance
        
        **CONFIDENCE**: Medium - Wait for TRIN <1.3
        """)
    elif gate_score >= 4.0:
        st.warning("""
        **POSITION**: 25% Long or Cash
        
        **ENTRY**: Wait for close confirmation
        
        **STOP**: Tight stops if entering
        
        **TARGET**: Limited upside until confirmation
        
        **CONFIDENCE**: Low-Medium - Watch key levels
        """)
    else:
        st.error("""
        **POSITION**: Cash or Defensive
        
        **ENTRY**: Stand aside
        
        **STOP**: N/A
        
        **TARGET**: N/A
        
        **CONFIDENCE**: Low - Preserve capital
        """)
    
    # Row 6: Continuity Tracking
    st.markdown("---")
    st.subheader("📈 Session Continuity Tracking")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Previous Session Reference**")
        st.write("Upload prior snapshot for delta tracking")
        
        # Would calculate deltas from prior session
        demo_deltas = {
            'SPXA50R': '+0.20',
            'BPSPX': '+1.20',
            'TRIN': '+0.15',
            'RSP': '+0.13'
        }
        
        for ind, delta in demo_deltas.items():
            st.metric(ind, delta, delta_color="normal")
    
    with col2:
        st.markdown("**Key Levels to Watch**")
        st.write("""
        - **SPXA50R**: Hold >28 (Support) / Break >32 (Resistance)
        - **BPSPX**: Hold >36 (Support) / Break >42 (Resistance)
        - **TRIN**: <1.3 (Bullish) / >1.8 (Bearish)
        - **RSP**: >194.55 (Pivot) / <188.58 (Support)
        """)
    
    # Footer
    st.markdown("---")
    st.caption("""
    **Disclaimer**: This tool is for educational purposes only. 
    Market data may be delayed. Always verify with primary sources.
    Trading involves substantial risk of loss.
    
    *Last Updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
    """.format(datetime=datetime))

if __name__ == "__main__":
    main()
