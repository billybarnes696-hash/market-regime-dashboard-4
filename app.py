import streamlit as st
import pandas as pd
import numpy as np
import zipfile
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Market Breadth Engine v2.0", layout="wide")

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

@st.cache_data
def load_historical_zip(zip_file):
    """Load historical CSV data from ZIP file"""
    data = {}
    with zipfile.ZipFile(zip_file) as z:
        for name in z.namelist():
            if name.endswith(".csv"):
                try:
                    df = pd.read_csv(z.open(name))
                    if "Date" in df.columns:
                        df["Date"] = pd.to_datetime(df["Date"])
                        df = df.set_index("Date")
                        if "Close" in df.columns:
                            symbol = name.replace(".csv", "").replace("$", "")
                            data[symbol] = df["Close"]
                except:
                    pass
    return data

@st.cache_data
def load_snapshot(file):
    """Load daily snapshot CSV"""
    return pd.read_csv(file)

@st.cache_data
def load_prior_upload(file):
    """Load prior upload for continuity tracking"""
    return pd.read_csv(file)

# ============================================================================
# ENHANCED GATE SCORE CALCULATION (v2.0)
# ============================================================================

def calculate_oscillator_consensus(indicators, oscillators):
    """
    Calculate oscillator consensus across all indicators
    Returns: % bullish, tally table
    """
    oscillator_weights = {
        'TSI': 0.25, 'MACD': 0.20, 'ADX': 0.20, 'MCCI': 0.15,
        'Stoch': 0.10, '%B': 0.05, 'ROC': 0.05
    }
    
    tally = []
    total_bullish = 0
    total_oscillators = 0
    
    for indicator in indicators.keys():
        if indicator in oscillators:
            osc_data = oscillators[indicator]
            bullish_count = 0
            
            # TSI Cross
            if osc_data.get('TSI_line', 0) > osc_data.get('TSI_signal', 0):
                bullish_count += 1
            # MACD Histogram
            if osc_data.get('MACD_hist', 0) > 0:
                bullish_count += 1
            # ADX +DI > -DI
            if osc_data.get('plus_DI', 0) > osc_data.get('minus_DI', 0):
                bullish_count += 1
            # MCCI Sign
            if osc_data.get('MCCI', 0) > 0:
                bullish_count += 1
            # Stoch %K > %D
            if osc_data.get('Stoch_K', 0) > osc_data.get('Stoch_D', 0):
                bullish_count += 1
            # %B Position (not overbought)
            if 0.20 < osc_data.get('%B', 0.50) < 0.80:
                bullish_count += 1
            # ROC Sign
            if osc_data.get('ROC', 0) > 0:
                bullish_count += 1
            
            tally.append({
                'Indicator': indicator,
                'Bullish': bullish_count,
                'Total': 7,
                '% Bullish': round(bullish_count / 7 * 100, 1)
            })
            
            total_bullish += bullish_count
            total_oscillators += 7
    
    consensus_pct = round(total_bullish / total_oscillators * 100, 1) if total_oscillators > 0 else 0
    return consensus_pct, pd.DataFrame(tally)

def calculate_trin_composite_score(trin_data):
    """
    Enhanced TRIN Gatekeeper Score (0-10)
    """
    score = 0
    
    # Daily Level (0-3 pts)
    trin_level = trin_data.get('level', 1.50)
    if trin_level < 1.0:
        score += 3
    elif trin_level < 1.3:
        score += 2
    elif trin_level < 1.6:
        score += 1
    
    # %B Position (0-3 pts)
    trin_pctb = trin_data.get('%B', 1.0)
    if trin_pctb < 0.80:
        score += 3
    elif trin_pctb < 1.20:
        score += 2
    
    # TSI Cross (0-2 pts)
    if trin_data.get('TSI_bullish', False):
        score += 2
    
    # ROC Velocity (0-2 pts)
    trin_roc = trin_data.get('ROC', 0)
    if 0 < trin_roc < 50:
        score += 2
    elif trin_roc >= 50:
        score += 1  # Overheating
    
    return min(score, 10)

def calculate_nymo_proxy_v2(nyad_data, spxadp_data, nyhl_data):
    """
    Enhanced NYMO Proxy v2.0
    NYMO = (NYAD Trend × 40%) + (SPXADP Thrust × 40%) + (NYHL Momentum × 20%)
    """
    # NYAD Trend Score (0-1.0)
    nyad_score = 0
    if nyad_data.get('rising', False):
        nyad_score += 0.30
    if nyad_data.get('%B', 0.50) < 0.60:
        nyad_score += 0.20
    if nyad_data.get('TSI_bullish', False):
        nyad_score += 0.30
    if nyad_data.get('MCCI', 0) > 0:
        nyad_score += 0.20
    
    # SPXADP Thrust Score (0-1.0)
    spxadp_score = 0
    if spxadp_data.get('level', 0) > 35:
        spxadp_score += 0.20
    if spxadp_data.get('TSI_bullish', False):
        spxadp_score += 0.30
    if spxadp_data.get('MACD_positive', False):
        spxadp_score += 0.20
    if spxadp_data.get('MCCI', 0) > 50:
        spxadp_score += 0.30
    
    # NYHL Momentum Score (0-0.3)
    nyhl_score = 0
    if nyhl_data.get('daily_positive', False):
        nyhl_score += 0.10
    if nyhl_data.get('5day_sum', 0) > 0:
        nyhl_score += 0.10
    if nyhl_data.get('renko_green', False):
        nyhl_score += 0.10
    
    # Calculate Proxy
    nymo_proxy = (nyad_score * 0.40) + (spxadp_score * 0.40) + (nyhl_score * 0.20)
    return round(nymo_proxy, 2)

def detect_market_regime(indicators, oscillators):
    """
    Detect market regime based on ADX
    Returns: 'Choppy', 'Transition', or 'Strong Trend'
    """
    avg_adx = np.mean([osc.get('ADX', 20) for osc in oscillators.values()])
    
    if avg_adx < 20:
        return 'Choppy'
    elif avg_adx > 40:
        return 'Strong Trend'
    else:
        return 'Transition'

def get_dynamic_weights(regime):
    """
    Get dynamic component weights based on market regime
    """
    if regime == 'Choppy':
        return {
            'Breadth Momentum': 0.30,
            'Distribution Filter': 0.25,
            'Price Confirmation': 0.15,
            'Ratio Leadership': 0.15,
            'Credit/Sentiment': 0.15
        }
    elif regime == 'Strong Trend':
        return {
            'Breadth Momentum': 0.20,
            'Distribution Filter': 0.15,
            'Price Confirmation': 0.25,
            'Ratio Leadership': 0.25,
            'Credit/Sentiment': 0.15
        }
    else:  # Transition
        return {
            'Breadth Momentum': 0.25,
            'Distribution Filter': 0.20,
            'Price Confirmation': 0.20,
            'Ratio Leadership': 0.20,
            'Credit/Sentiment': 0.15
        }

def calculate_gate_score_v2(indicators, oscillators, trin_data, nyad_data, spxadp_data, nyhl_data, prior_data=None):
    """
    Enhanced Gate Score Calculation v2.0
    Returns: Score (0-10), component breakdown, commentary
    """
    # Detect regime and get dynamic weights
    regime = detect_market_regime(indicators, oscillators)
    weights = get_dynamic_weights(regime)
    
    # Calculate oscillator consensus
    osc_consensus, osc_tally = calculate_oscillator_consensus(indicators, oscillators)
    
    # Calculate TRIN composite score
    trin_score = calculate_trin_composite_score(trin_data)
    
    # Calculate NYMO Proxy
    nymo_proxy = calculate_nymo_proxy_v2(nyad_data, spxadp_data, nyhl_data)
    
    # Component Scores (0-10 scale)
    components = {}
    
    # 1. Breadth Momentum (SPXA50R, BPSPX, BPNYA)
    breadth_score = 0
    if indicators.get('SPXA50R', 0) > 40:
        breadth_score += 4
    elif indicators.get('SPXA50R', 0) > 30:
        breadth_score += 3
    elif indicators.get('SPXA50R', 0) > 28:
        breadth_score += 2
    else:
        breadth_score += 1
    
    if indicators.get('BPSPX', 0) > 50:
        breadth_score += 3
    elif indicators.get('BPSPX', 0) > 40:
        breadth_score += 2
    elif indicators.get('BPSPX', 0) > 35:
        breadth_score += 1
    
    if indicators.get('BPNYA', 0) > 50:
        breadth_score += 3
    elif indicators.get('BPNYA', 0) > 40:
        breadth_score += 2
    
    components['Breadth Momentum'] = min(breadth_score, 10)
    
    # 2. Distribution Filter (TRIN, SPXADP)
    dist_score = 0
    if trin_score >= 7:
        dist_score += 5
    elif trin_score >= 4:
        dist_score += 3
    elif trin_score >= 2:
        dist_score += 1
    
    if indicators.get('SPXADP', 0) > 50:
        dist_score += 3
    elif indicators.get('SPXADP', 0) > 35:
        dist_score += 2
    elif indicators.get('SPXADP', 0) > 20:
        dist_score += 1
    
    components['Distribution Filter'] = min(dist_score, 10)
    
    # 3. Price Confirmation (RSP, SPX, URSP)
    price_score = 0
    if indicators.get('RSP', 0) > 194.55:  # Above pivot
        price_score += 4
    elif indicators.get('RSP', 0) > 188.58:  # Above EMAENV lower
        price_score += 2
    
    if indicators.get('SPX', 0) > 6582.15:  # Above pivot
        price_score += 3
    
    if indicators.get('URSP', 0) > 41.77:  # Above pivot
        price_score += 3
    
    components['Price Confirmation'] = min(price_score, 10)
    
    # 4. Ratio Leadership (RSP:SPY, IWM:SPY, XLF:SPY)
    ratio_score = 0
    if indicators.get('RSP:SPY', 0) > 0.302:
        ratio_score += 4
    elif indicators.get('RSP:SPY', 0) > 0.294:
        ratio_score += 2
    
    if indicators.get('IWM:SPY', 0) > 0.3943:
        ratio_score += 3
    elif indicators.get('IWM:SPY', 0) > 0.3775:
        ratio_score += 2
    
    if indicators.get('XLF:SPY', 0) > 0.0779:
        ratio_score += 3
    elif indicators.get('XLF:SPY', 0) > 0.0751:
        ratio_score += 1
    
    components['Ratio Leadership'] = min(ratio_score, 10)
    
    # 5. Credit/Sentiment (HYG:IEF, HYG:TLT, VIX, CPCE)
    credit_score = 0
    if indicators.get('VIX', 99) < 20:
        credit_score += 3
    elif indicators.get('VIX', 99) < 25:
        credit_score += 2
    
    if indicators.get('CPCE', 0) < 0.60:
        credit_score += 3
    elif indicators.get('CPCE', 0) < 0.70:
        credit_score += 2
    
    if indicators.get('HYG:IEF', 0) > 0.840:
        credit_score += 2
    
    if indicators.get('HYG:TLT', 0) > 0.932:
        credit_score += 2
    
    components['Credit/Sentiment'] = min(credit_score, 10)
    
    # Calculate weighted score
    final_score = 0
    for component, score in components.items():
        final_score += score * weights[component]
    
    final_score = round(min(max(final_score, 0), 10), 1)
    
    # Continuity tracking (if prior data available)
    delta = None
    if prior_data is not None and 'Gate Score' in prior_data:
        delta = round(final_score - prior_data['Gate Score'], 1)
    
    # Commentary
    if final_score >= 7.0:
        action = "🟢 Standard Entry"
        position = "50-75%"
    elif final_score >= 5.0:
        action = "🟡 Scale-In"
        position = "25-50%"
    elif final_score >= 3.0:
        action = "🟡 Watch Only"
        position = "0-25%"
    else:
        action = "🔴 Stand Aside"
        position = "0%"
    
    return {
        'Gate Score': final_score,
        'Delta': delta,
        'Action': action,
        'Position': position,
        'Components': components,
        'Weights': weights,
        'Regime': regime,
        'Oscillator Consensus': osc_consensus,
        'Oscillator Tally': osc_tally,
        'TRIN Score': trin_score,
        'NYMO Proxy': nymo_proxy
    }

def get_position_sizing(gate_score):
    """
    Dynamic position sizing based on Gate Score
    """
    if gate_score >= 7.0:
        return {
            'Position': '75-100%',
            'Stop': '2x ATR or S1 pivot',
            'Target': 'R2/R3 or trail below 20-EMA',
            'Conviction': 'High'
        }
    elif gate_score >= 5.0:
        return {
            'Position': '50%',
            'Stop': '1.5x ATR or EMAENV lower',
            'Target': 'R1/R2',
            'Conviction': 'Medium-High'
        }
    elif gate_score >= 3.0:
        return {
            'Position': '25%',
            'Stop': '1x ATR or S2 pivot',
            'Target': 'R1',
            'Conviction': 'Low'
        }
    else:
        return {
            'Position': '0%',
            'Stop': 'N/A',
            'Target': 'N/A',
            'Conviction': 'Avoid'
        }

def get_targets_triggers(indicators):
    """
    Generate specific targets and triggers for key indicators
    """
    targets = {
        'RSP': {
            'Current': indicators.get('RSP', 0),
            'Bullish Target': '>194.55 (pivot), >198.25 (EMAENV)',
            'Bearish Invalidation': '<188.58 (EMAENV lower)',
            'Confirmation Trigger': 'Close >194.55 + TSI hold'
        },
        'BPSPX': {
            'Current': indicators.get('BPSPX', 0),
            'Bullish Target': '>42.20 (pivot), %B >0.60',
            'Bearish Invalidation': '<34.40 (intraday low)',
            'Confirmation Trigger': 'Close >40 + Stoch cross hold'
        },
        'SPXA50R': {
            'Current': indicators.get('SPXA50R', 0),
            'Bullish Target': '>35.73 (pivot), >32.04 (EMAENV)',
            'Bearish Invalidation': '<26.20 (intraday low)',
            'Confirmation Trigger': 'Hold >28 into close'
        },
        'TRIN': {
            'Current': indicators.get('TRIN', 0),
            'Bullish Target': '<1.20 (neutral), <1.10 (bullish)',
            'Bearish Invalidation': '>1.90 (extreme distribution)',
            'Confirmation Trigger': 'Close <1.40 + TSI hold'
        },
        'SPXADP': {
            'Current': indicators.get('SPXADP', 0),
            'Bullish Target': '>50.00 (broad participation)',
            'Bearish Invalidation': '<20.13 (pivot breakdown)',
            'Confirmation Trigger': 'Stabilize >35 into close'
        },
        'RSP:SPY': {
            'Current': indicators.get('RSP:SPY', 0),
            'Bullish Target': '>0.302 (R1), >0.310 (confirmation)',
            'Bearish Invalidation': '<0.287 (EMAENV lower)',
            'Confirmation Trigger': 'Close >0.300 + TSI hold'
        }
    }
    return targets

# ============================================================================
# STREAMLIT UI
# ============================================================================

st.title("📊 Market Breadth Decision Engine v2.0")
st.markdown("### Enhanced Multi-Factor Market Regime Framework")

# Sidebar
st.sidebar.header("📁 Upload Data")
hist_zip = st.sidebar.file_uploader("Historical ZIP", type="zip")
daily_csv = st.sidebar.file_uploader("Daily Snapshot CSV", type="csv")
prior_csv = st.sidebar.file_uploader("Prior Upload (for continuity)", type="csv")

# Load data
historical_data = {}
snapshot = None
prior_data = None

if hist_zip:
    historical_data = load_historical_zip(hist_zip)
    st.sidebar.success(f"✅ {len(historical_data)} historical series loaded")

if daily_csv:
    snapshot = load_snapshot(daily_csv)
    st.sidebar.success("✅ Snapshot loaded")

if prior_csv:
    prior_data = load_prior_upload(prior_csv)
    st.sidebar.success("✅ Prior upload loaded for continuity tracking")

# Default indicators (can be overridden by snapshot)
indicators = {
    'SPXA50R': 29.80,
    'BPSPX': 38.20,
    'BPNYA': 44.49,
    'TRIN': 1.65,
    'SPXADP': 28.20,
    'RSP': 192.85,
    'SPX': 6585.79,
    'VIX': 24.56,
    'CPCE': 0.56,
    'RSP:SPY': 0.294,
    'IWM:SPY': 0.3813,
    'XLF:SPY': 0.0754,
    'HYG:IEF': 0.835,
    'HYG:TLT': 0.920
}

# Override with snapshot data
if snapshot is not None and "Symbol" in snapshot.columns:
    for _, row in snapshot.iterrows():
        s = str(row["Symbol"]).replace("$", "")
        if s in indicators:
            indicators[s] = float(row.get("Close", row.get("Last", 0)))

# Default oscillator data (would come from snapshot in production)
oscillators = {
    'SPXA50R': {'TSI_line': -39.39, 'TSI_signal': -42.61, 'MACD_hist': 0.532, 'ADX': 49.42, 
                'plus_DI': 15.87, 'minus_DI': 33.45, 'MCCI': -35.59, 'Stoch_K': 25.95, 
                'Stoch_D': 15.28, '%B': 0.41, 'ROC': -14.37},
    'BPSPX': {'TSI_line': -37.84, 'TSI_signal': -39.78, 'MACD_hist': -0.154, 'ADX': 42.22,
              'plus_DI': 26.44, 'minus_DI': 49.81, 'MCCI': -73.41, 'Stoch_K': 20.40,
              'Stoch_D': 13.21, '%B': 0.34, 'ROC': -11.98},
    'TRIN': {'TSI_line': 12.55, 'TSI_signal': -1.17, 'MACD_hist': 0.059, 'ADX': 37.67,
             'plus_DI': 32.83, 'minus_DI': 10.86, 'MCCI': 189.19, 'Stoch_K': 47.99,
             'Stoch_D': 39.91, '%B': 1.30, 'ROC': 101.02, 'TSI_bullish': True},
    'RSP': {'TSI_line': -22.67, 'TSI_signal': -24.20, 'MACD_hist': -0.011, 'ADX': 40.53,
            'plus_DI': 16.12, 'minus_DI': 30.04, 'MCCI': -19.14, 'Stoch_K': 39.57,
            'Stoch_D': 24.04, '%B': 0.46, 'ROC': -0.66}
}

# NYAD, SPXADP, NYHL data for NYMO Proxy
nyad_data = {
    'rising': True, '%B': 0.46, 'TSI_bullish': True, 'MCCI': -11.92
}
spxadp_data = {
    'level': 28.20, 'TSI_bullish': True, 'MACD_positive': True, 'MCCI': 58.85
}
nyhl_data = {
    'daily_positive': True, '5day_sum': 50, 'renko_green': False
}

trin_data = {
    'level': indicators.get('TRIN', 1.65),
    '%B': oscillators.get('TRIN', {}).get('%B', 1.30),
    'TSI_bullish': oscillators.get('TRIN', {}).get('TSI_bullish', False),
    'ROC': oscillators.get('TRIN', {}).get('ROC', 101.02)
}

# Calculate Gate Score
result = calculate_gate_score_v2(
    indicators, oscillators, trin_data, nyad_data, spxadp_data, nyhl_data, prior_data
)

# Get position sizing
position_info = get_position_sizing(result['Gate Score'])

# Get targets/triggers
targets = get_targets_triggers(indicators)

# ============================================================================
# DASHBOARD
# ============================================================================

# Top Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="🎯 Gate Score",
        value=f"{result['Gate Score']}/10",
        delta=f"{result['Delta']}" if result['Delta'] else None,
        delta_color="normal" if result['Delta'] is None else ("normal" if result['Delta'] >= 0 else "inverse")
    )

with col2:
    st.metric(
        label="📊 Oscillator Consensus",
        value=f"{result['Oscillator Consensus']}%",
        help="% of oscillators showing bullish signals"
    )

with col3:
    st.metric(
        label="🔍 TRIN Composite",
        value=f"{result['TRIN Score']}/10",
        help="TRIN gatekeeper score (≥7 = distribution abating)"
    )

with col4:
    st.metric(
        label="📈 NYMO Proxy",
        value=result['NYMO Proxy'],
        help=">1.5 = Strong thrust, 0.5-1.5 = Improving, <-0.5 = Deteriorating"
    )

st.divider()

# Action Panel
col1, col2, col3 = st.columns(3)

with col1:
    st.info(f"**Action:** {result['Action']}")
    st.success(f"**Position Size:** {position_info['Position']}")
    st.write(f"**Conviction:** {position_info['Conviction']}")

with col2:
    st.write(f"**Market Regime:** {result['Regime']}")
    st.write(f"**Stop Loss:** {position_info['Stop']}")
    st.write(f"**Target:** {position_info['Target']}")

with col3:
    st.write("### Component Breakdown")
    for component, score in result['Components'].items():
        weight = result['Weights'][component]
        contribution = round(score * weight, 2)
        st.write(f"{component}: {score}/10 × {weight*100:.0f}% = {contribution}")

st.divider()

# Oscillator Consensus Table
st.subheader("📊 Oscillator Consensus Table")
if 'Oscillator Tally' in result and not result['Oscillator Tally'].empty:
    st.dataframe(result['Oscillator Tally'], use_container_width=True)
else:
    st.write("Oscillator data not available")

st.divider()

# Targets & Triggers
st.subheader("🎯 Specific Targets & Triggers")
targets_df = pd.DataFrame(targets).T
st.dataframe(targets_df, use_container_width=True)

st.divider()

# Visualizations
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Gate Score Distribution")
    scores = np.random.normal(result['Gate Score'], 1.5, 1000)
    scores = np.clip(scores, 0, 10)
    fig = go.Figure()
    fig.add_histogram(x=scores, nbinsx=40, marker_color='lightblue')
    fig.add_vline(x=result['Gate Score'], line_color="red", line_width=3)
    fig.update_layout(title="Gate Score Distribution", xaxis_title="Score", yaxis_title="Frequency")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📊 Component Weights")
    fig = go.Figure(data=[
        go.Pie(
            labels=list(result['Weights'].keys()),
            values=[v*100 for v in result['Weights'].values()],
            hole=0.3
        )
    ])
    fig.update_layout(title="Dynamic Component Weights")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# Continuity Tracking
if prior_data is not None:
    st.subheader("📋 Continuity Tracking vs. Prior Upload")
    delta_data = {
        'Indicator': ['SPXA50R', 'BPSPX', 'TRIN', 'RSP', 'Gate Score'],
        'Prior': [
            prior_data.get('SPXA50R', 0),
            prior_data.get('BPSPX', 0),
            prior_data.get('TRIN', 0),
            prior_data.get('RSP', 0),
            prior_data.get('Gate Score', 0)
        ],
        'Current': [
            indicators.get('SPXA50R', 0),
            indicators.get('BPSPX', 0),
            indicators.get('TRIN', 0),
            indicators.get('RSP', 0),
            result['Gate Score']
        ]
    }
    delta_df = pd.DataFrame(delta_data)
    delta_df['Delta'] = delta_df['Current'] - delta_df['Prior']
    delta_df['Delta'] = delta_df['Delta'].round(2)
    st.dataframe(delta_df, use_container_width=True)

st.divider()

# Export
st.subheader("📤 Export Report")
if st.button("Generate CSV Report"):
    report_data = {
        'Timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        'Gate Score': [result['Gate Score']],
        'Action': [result['Action']],
        'Position': [position_info['Position']],
        'Regime': [result['Regime']],
        'Oscillator Consensus': [result['Oscillator Consensus']],
        'TRIN Score': [result['TRIN Score']],
        'NYMO Proxy': [result['NYMO Proxy']]
    }
    for indicator, value in indicators.items():
        report_data[f'{indicator}_Value'] = [value]
    
    report_df = pd.DataFrame(report_data)
    csv = report_df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"breadth_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

st.caption("⚠️ Upload historical ZIP + daily snapshot + prior upload for full analysis. This is a decision support tool, not investment advice.")
