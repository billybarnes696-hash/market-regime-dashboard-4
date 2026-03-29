import io
import json
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

# -----------------------------
# App config + styling
# -----------------------------
st.set_page_config(page_title="Breadth Sweet Spot Cluster Engine v2", layout="wide", page_icon="📈")

CUSTOM_CSS = """
<style>
:root{
  --bg:#0b1020;
  --panel:#111936;
  --panel-2:#162246;
  --text:#eaf0ff;
  --muted:#9cb0df;
  --green:#22c55e;
  --yellow:#f59e0b;
  --red:#ef4444;
  --blue:#38bdf8;
  --purple:#a78bfa;
}
.block-container{padding-top:1rem;padding-bottom:2rem;}
.main-title{padding:1rem 1.2rem;border-radius:18px;background:linear-gradient(135deg, rgba(56,189,248,.16), rgba(167,139,250,.16));border:1px solid rgba(148,163,184,.20);margin-bottom:1rem;}
.soft-card{background:linear-gradient(180deg, rgba(17,25,54,.96), rgba(10,17,38,.98));border:1px solid rgba(148,163,184,.24);border-radius:18px;padding:1rem;box-shadow:0 10px 35px rgba(0,0,0,.22);}
.score-card{min-height:160px;display:flex;flex-direction:column;justify-content:space-between;}
.score-title{color:#bcd0ff;font-size:1.02rem;font-weight:800;letter-spacing:.02em;}
.score-value{font-size:3rem;line-height:1.0;font-weight:950;color:#fff;margin:.3rem 0;letter-spacing:-.03em;}
.score-subtitle{font-size:.98rem;font-weight:700;color:#cfe0ff;}
.score-bar{width:100%;height:12px;border-radius:999px;background:rgba(255,255,255,.1);overflow:hidden;border:1px solid rgba(255,255,255,.08);margin-top:.65rem;}
.score-fill{height:100%;border-radius:999px;}
.score-fill-green{background:linear-gradient(90deg,#22c55e,#4ade80);} 
.score-fill-yellow{background:linear-gradient(90deg,#f59e0b,#fbbf24);} 
.score-fill-red{background:linear-gradient(90deg,#ef4444,#f87171);} 
.score-fill-blue{background:linear-gradient(90deg,#38bdf8,#60a5fa);} 
.kpi-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.75rem;margin-top:.85rem;}
.kpi-box{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.10);border-radius:14px;padding:.8rem .85rem;}
.kpi-label{color:#9fb2de;font-size:.9rem;font-weight:700;}
.kpi-value{color:#fff;font-size:1.75rem;font-weight:900;line-height:1.05;margin-top:.15rem;}
.action-box{border-radius:16px;padding:.85rem 1rem;margin:.5rem 0;border:1px solid rgba(255,255,255,.10);}
.action-existing{background:rgba(56,189,248,.10);} 
.action-new{background:rgba(245,158,11,.10);} 
.action-add{background:rgba(34,197,94,.10);} 
.pill{display:inline-block;padding:.3rem .6rem;border-radius:999px;font-size:.82rem;font-weight:700;border:1px solid rgba(255,255,255,.12);margin-right:.35rem;}
.pill-green{background:rgba(34,197,94,.16);color:#bbf7d0;} 
.pill-yellow{background:rgba(245,158,11,.16);color:#fde68a;} 
.pill-red{background:rgba(239,68,68,.16);color:#fecaca;} 
.pill-blue{background:rgba(56,189,248,.16);color:#bae6fd;} 
.pill-purple{background:rgba(167,139,250,.16);color:#ddd6fe;} 
.small-muted{color:#93a4cc;font-size:.88rem;}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown(
    """
    <div class="main-title">
      <div style="font-size:1.55rem;font-weight:900;">📈 Breadth Sweet Spot Cluster Engine v2</div>
      <div class="small-muted">Empirical sweet spots + clustering + canary overlay + walk-forward validation.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Constants / storage
# -----------------------------
APP_DIR = Path("breadth_cluster_store_v2")
APP_DIR.mkdir(exist_ok=True)
HIST_DAILY_PATH = APP_DIR / "daily_history.parquet"
HIST_WEEKLY_PATH = APP_DIR / "weekly_history.parquet"
MODEL_META_PATH = APP_DIR / "model_meta.json"
SWEET_SPOT_PATH = APP_DIR / "sweet_spots.json"
CLUSTER_INFO_PATH = APP_DIR / "cluster_info.json"
UPLOAD_HISTORY_PATH = APP_DIR / "upload_history.csv"
SNAPSHOT_DIR = APP_DIR / "snapshots"
SNAPSHOT_DIR.mkdir(exist_ok=True)

FEATURES = [
    "$BPSPX", "$BPSPX_%B", "$BPNYA", "$OEXA200R", "$SPXA50R", "$NYMO", "$NYSI", "$CPCE",
    "$NYHL", "$NYAD", "$SPXADP", "RSP:SPY"
]
KEY_FEATURES_FOR_SCORING = ["$BPSPX_%B", "$BPNYA", "$OEXA200R", "$SPXA50R", "$NYMO", "$NYSI", "$CPCE", "$NYHL", "RSP:SPY"]
WEEKLY_FEATURES = ["$BPSPX", "$SPXA50R", "$NYSI", "$OEXA200R", "RSP"]
CANARY_FEATURES = ["RSP:SPY", "SMH:SPY", "IWM:SPY", "XLF:SPY", "XLY:SPY", "HYG:SHY", "SPXS:SVOL", "SPY:VXX"]
OUTCOME_WINDOWS = {"bounce": 10, "repair": 20, "regime": 60}
MIN_FEATURE_OBS = 40
EASTERN = "America/New_York"

# -----------------------------
# Helpers
# -----------------------------
def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return np.nan


def fmt_num(v, d=2):
    if pd.isna(v):
        return "n/a"
    return f"{float(v):.{d}f}"


def save_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2))


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def metric_color(score, max_score):
    ratio = 0 if max_score == 0 or pd.isna(score) else score / max_score
    if ratio >= 0.7:
        return "green"
    if ratio >= 0.5:
        return "yellow"
    if ratio >= 0.3:
        return "blue"
    return "red"


def score_card(title: str, value: float, max_score: float, subtitle: str = ""):
    color = metric_color(value, max_score)
    pct = 0 if pd.isna(value) or max_score == 0 else max(0, min(100, 100 * value / max_score))
    st.markdown(
        f"""
        <div class="soft-card score-card">
          <div>
            <div class="score-title">{title}</div>
            <div class="score-value">{0 if pd.isna(value) else value:.0f}</div>
            <div class="score-subtitle">{subtitle}</div>
          </div>
          <div class="score-bar"><div class="score-fill score-fill-{color}" style="width:{pct:.0f}%"></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def market_phase(ts: Optional[pd.Timestamp] = None) -> str:
    now = pd.Timestamp.now(tz=EASTERN) if ts is None else pd.Timestamp(ts)
    if now.tzinfo is None:
        now = now.tz_localize(EASTERN)
    hhmm = now.hour + now.minute / 60
    if hhmm < 11:
        return "Morning"
    if hhmm < 14:
        return "Midday"
    if hhmm < 16:
        return "Power Hour"
    if hhmm < 18:
        return "Post Close"
    return "Official EOD"


def warning_box(messages: List[str], title: str = "Warnings"):
    if messages:
        st.warning(f"**{title}:**\n\n- " + "\n- ".join(messages))


# -----------------------------
# Indicators
# -----------------------------
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1 / period, adjust=False).mean()
    ma_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


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


def tsi(series: pd.Series, long_: int = 25, short_: int = 13, signal: int = 7):
    m = series.diff()
    a = m.abs()
    m1 = ema(ema(m, long_), short_)
    a1 = ema(ema(a, long_), short_)
    tsi_val = 100 * (m1 / a1.replace(0, np.nan))
    sig = ema(tsi_val, signal)
    return tsi_val, sig


def add_indicator_features(hist: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for sym, g in hist.groupby("symbol", sort=False):
        g = g.sort_values("date").copy()
        g["rsi14"] = rsi(g["close"], 14)
        g["cci20"] = cci(g["high"], g["low"], g["close"], 20)
        g["pct_b20"] = percent_b(g["close"], 20, 2.0)
        g["roc3"] = roc(g["close"], 3)
        g["slope3"] = g["close"] - g["close"].shift(3)
        g["tsi_fast"], g["tsi_fast_sig"] = tsi(g["close"], 4, 2, 4)
        g["ma20"] = g["close"].rolling(20).mean()
        g["ma50"] = g["close"].rolling(50).mean()
        frames.append(g)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# -----------------------------
# Parsing
# -----------------------------
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
        nums = [safe_float(x) for x in parts[1:6]]
        if all(pd.isna(x) for x in nums[:4]):
            continue
        rows.append({"date": dt, "open": nums[0], "high": nums[1], "low": nums[2], "close": nums[3], "volume": nums[4]})
    if not rows:
        raise ValueError("No rows parsed from StockCharts CSV")
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def symbol_from_filename(name: str) -> Tuple[str, str]:
    stem = Path(name).stem.strip().lower()
    timeframe = "weekly" if stem.endswith(" w") or stem.endswith("_w") else "daily"
    stem = stem.replace(" w", "").replace("_w", "")
    mapping = {
        "rsp": "RSP", "ursp": "URSP", "spy": "SPY", "vxx": "VXX",
        "_bpspx": "$BPSPX", "bpspx": "$BPSPX",
        "_bpnya": "$BPNYA", "bpnya": "$BPNYA",
        "_oexa200r": "$OEXA200R", "oexa200r": "$OEXA200R",
        "_spxa50r": "$SPXA50R", "spxa50r": "$SPXA50R",
        "_nymo": "$NYMO", "nymo": "$NYMO",
        "_nysi": "$NYSI", "nysi": "$NYSI",
        "_cpce": "$CPCE", "cpce": "$CPCE",
        "_nyhl": "$NYHL", "nyhl": "$NYHL",
        "_nyad": "$NYAD", "nyad": "$NYAD",
        "_spxadp": "$SPXADP", "spxadp": "$SPXADP",
        "_hyg_ief": "HYG:IEF", "hyg_ief": "HYG:IEF",
        "_hyg_shy": "HYG:SHY", "hyg_shy": "HYG:SHY",
        "_rsp_spy": "RSP:SPY", "rsp_spy": "RSP:SPY",
        "_smh_spy": "SMH:SPY", "smh_spy": "SMH:SPY",
        "_iwm_spy": "IWM:SPY", "iwm_spy": "IWM:SPY",
        "_xlf_spy": "XLF:SPY", "xlf_spy": "XLF:SPY",
        "_xly_spy": "XLY:SPY", "xly_spy": "XLY:SPY",
        "_spxs_svol": "SPXS:SVOL", "spxs_svol": "SPXS:SVOL",
        "_spy_vxx": "SPY:VXX", "spy_vxx": "SPY:VXX",
    }
    return mapping.get(stem, stem.upper()), timeframe


def parse_stockcharts_zip(file_bytes: bytes) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    daily, weekly, issues = [], [], []
    try:
        zf = zipfile.ZipFile(io.BytesIO(file_bytes))
    except Exception as e:
        raise ValueError(f"Zip file could not be opened: {e}")
    with zf:
        for name in zf.namelist():
            if name.endswith("/") or not name.lower().endswith(".csv"):
                continue
            try:
                content = zf.read(name)
                df = parse_stockcharts_csv(content)
                sym, tf = symbol_from_filename(name)
                df["symbol"] = sym
                (weekly if tf == "weekly" else daily).append(df)
            except Exception as e:
                issues.append(f"Skipped {name}: {e}")
    if not daily:
        raise ValueError("No daily CSVs were parsed from the zip file.")
    daily_df = pd.concat(daily, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    weekly_df = pd.concat(weekly, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True) if weekly else pd.DataFrame()
    return daily_df, weekly_df, issues


def parse_realtime_snapshot(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    if "Symbol" not in df.columns:
        raise ValueError("Realtime snapshot must contain a Symbol column.")
    close_candidates = ["Close", "Last", "Price", "Current", "Value", "Daily Close", "Close Price"]
    close_col = next((c for c in close_candidates if c in df.columns), None)
    if close_col is None:
        raise ValueError("Realtime snapshot must include a close-like price column.")
    df = df.copy()
    df["Symbol"] = df["Symbol"].astype(str).str.strip()
    df["Close"] = pd.to_numeric(df[close_col], errors="coerce")
    pct_candidates = ["Daily PctChange(1,Daily Close)", "% Change", "Pct Change", "Change %", "Daily Change %"]
    pct_col = next((c for c in pct_candidates if c in df.columns), None)
    df["PctChange"] = pd.to_numeric(df[pct_col], errors="coerce") if pct_col else np.nan
    return df[["Symbol", "Close", "PctChange"]]

# -----------------------------
# Wide feature frame
# -----------------------------
def wide_from_hist(hist_feat: pd.DataFrame) -> pd.DataFrame:
    if hist_feat.empty:
        return pd.DataFrame()
    value_cols = ["close", "pct_b20", "rsi14", "cci20", "roc3", "slope3", "tsi_fast"]
    pivoted = []
    for col in value_cols:
        p = hist_feat.pivot(index="date", columns="symbol", values=col)
        if col == "close":
            p.columns = [str(c) for c in p.columns]
        elif col == "pct_b20":
            p.columns = [f"{c}_%B" for c in p.columns]
        elif col == "rsi14":
            p.columns = [f"{c}_RSI14" for c in p.columns]
        elif col == "cci20":
            p.columns = [f"{c}_CCI20" for c in p.columns]
        elif col == "roc3":
            p.columns = [f"{c}_ROC3" for c in p.columns]
        elif col == "slope3":
            p.columns = [f"{c}_SLOPE3" for c in p.columns]
        elif col == "tsi_fast":
            p.columns = [f"{c}_TSI" for c in p.columns]
        pivoted.append(p)
    wide = pd.concat(pivoted, axis=1).sort_index()
    return wide


def current_snapshot_from_hist(hist_feat: pd.DataFrame, realtime_df: Optional[pd.DataFrame] = None) -> Tuple[Dict[str, float], Dict[str, float], pd.Timestamp]:
    wide = wide_from_hist(hist_feat)
    if wide.empty:
        return {}, {}, pd.NaT
    latest_row = wide.iloc[-1].copy()
    prev_row = wide.iloc[-2].copy() if len(wide) > 1 else latest_row.copy()
    latest_date = wide.index[-1]
    if realtime_df is not None and not realtime_df.empty:
        rt_map = dict(zip(realtime_df["Symbol"], realtime_df["Close"]))
        for sym, px in rt_map.items():
            if sym in hist_feat["symbol"].unique():
                g = hist_feat[hist_feat["symbol"] == sym].sort_values("date").copy()
                if len(g) >= 20:
                    g.iloc[-1, g.columns.get_loc("close")] = px
                    g.iloc[-1, g.columns.get_loc("high")] = max(g.iloc[-1]["high"], px)
                    g.iloc[-1, g.columns.get_loc("low")] = min(g.iloc[-1]["low"], px)
                    if len(g) > 1:
                        g.iloc[-1, g.columns.get_loc("open")] = g.iloc[-2]["close"]
                    g = add_indicator_features(g)
                    last = g.sort_values("date").iloc[-1]
                    latest_row[sym] = safe_float(last["close"])
                    latest_row[f"{sym}_%B"] = safe_float(last["pct_b20"])
                    latest_row[f"{sym}_RSI14"] = safe_float(last["rsi14"])
                    latest_row[f"{sym}_CCI20"] = safe_float(last["cci20"])
                    latest_row[f"{sym}_ROC3"] = safe_float(last["roc3"])
                    latest_row[f"{sym}_SLOPE3"] = safe_float(last["slope3"])
                    latest_row[f"{sym}_TSI"] = safe_float(last["tsi_fast"])
    return latest_row.to_dict(), prev_row.to_dict(), latest_date

# -----------------------------
# Outcome definitions + sweet spots
# -----------------------------
def forward_metrics(rsp: pd.Series, horizon: int) -> pd.DataFrame:
    vals = rsp.to_numpy(dtype=float)
    n = len(vals)
    if n == 0:
        return pd.DataFrame(index=rsp.index)
    future_cols = [np.roll(vals, -(i + 1)) / vals - 1 for i in range(horizon)]
    arr = np.column_stack(future_cols)
    if horizon > 0:
        arr[-horizon:, :] = np.nan
    out = pd.DataFrame(index=rsp.index)
    out[f"ret_{horizon}"] = arr[:, horizon - 1]
    out[f"max_{horizon}"] = np.nanmax(arr, axis=1)
    out[f"min_{horizon}"] = np.nanmin(arr, axis=1)
    out.iloc[-horizon:, :] = np.nan
    return out


def label_outcomes(wide_daily: pd.DataFrame) -> pd.DataFrame:
    if "RSP" not in wide_daily.columns:
        raise ValueError("Historical daily baseline must include RSP for outcome labeling.")
    rsp = wide_daily["RSP"].dropna().copy()
    metrics = [forward_metrics(rsp, h) for h in [10, 20, 60]]
    df = pd.concat([wide_daily, *metrics], axis=1)
    df["bounce"] = (df["max_10"] >= 0.03) & (df["min_10"] > -0.03)
    df["repair"] = (df["ret_20"] >= 0.04) & (df["min_20"] > -0.05)
    df["regime"] = (df["ret_60"] >= 0.08) & (df["ret_20"] >= 0.03) & (df["min_60"] > -0.08)
    return df


def summarize_feature_bands(df: pd.DataFrame, label: str, features: List[str]) -> Dict[str, Dict[str, float]]:
    hit = df[df[label] == True]
    result = {}
    for feat in features:
        if feat not in hit.columns:
            continue
        s = pd.to_numeric(hit[feat], errors="coerce").dropna()
        if len(s) < 10:
            continue
        result[feat] = {
            "median": float(s.median()),
            "q25": float(s.quantile(0.25)),
            "q75": float(s.quantile(0.75)),
            "mean": float(s.mean()),
            "std": float(0 if pd.isna(s.std()) else s.std()),
            "count": int(s.shape[0]),
        }
    return result


def best_combo_zones(df: pd.DataFrame, label: str, features: List[str]) -> pd.DataFrame:
    base_rate = float(df[label].mean()) if len(df) else np.nan
    rows = []
    usable = [f for f in features if f in df.columns]
    from itertools import combinations
    for combo_len in [2, 3]:
        for combo in combinations(usable, combo_len):
            mask = pd.Series(True, index=df.index)
            desc = []
            for feat in combo:
                hit_s = pd.to_numeric(df.loc[df[label] == True, feat], errors="coerce").dropna()
                if len(hit_s) < 10:
                    mask &= False
                    continue
                lo, hi = hit_s.quantile(0.25), hit_s.quantile(0.75)
                cur = pd.to_numeric(df[feat], errors="coerce")
                mask &= cur.between(lo, hi, inclusive="both")
                desc.append(f"{feat}∈[{lo:.3f},{hi:.3f}]")
            sub = df[mask]
            if len(sub) < 25:
                continue
            hit_rate = float(sub[label].mean())
            lift = hit_rate / base_rate if base_rate and not pd.isna(base_rate) else np.nan
            rows.append({"label": label, "combo": " | ".join(combo), "zone": "; ".join(desc), "samples": int(len(sub)), "hit_rate": hit_rate, "base_rate": base_rate, "lift": lift})
    out = pd.DataFrame(rows)
    return out.sort_values(["lift", "hit_rate", "samples"], ascending=[False, False, False]).head(20).reset_index(drop=True) if not out.empty else out

# -----------------------------
# Clustering + validation
# -----------------------------
@dataclass
class ClusterArtifacts:
    scaler_mean: List[float]
    scaler_scale: List[float]
    features: List[str]
    centroids: List[List[float]]
    cluster_names: Dict[str, str]
    cluster_stats: Dict[str, Dict[str, float]]
    silhouette: Optional[float] = None
    stability: Optional[float] = None


def assign_cluster_names(stats_df: pd.DataFrame) -> Dict[int, str]:
    names = {}
    for idx, row in stats_df.iterrows():
        bounce = row.get("bounce_rate", 0)
        repair = row.get("repair_rate", 0)
        regime = row.get("regime_rate", 0)
        if regime >= max(repair, bounce) and regime > 0.30:
            names[idx] = "Durable regime"
        elif repair >= max(regime, bounce) and repair > 0.22:
            names[idx] = "Repair cluster"
        elif bounce >= max(regime, repair) and bounce > 0.35:
            names[idx] = "Bounce cluster"
        elif row.get("$BPSPX_%B_median", np.nan) < 0.15:
            names[idx] = "Capitulation / washout"
        else:
            names[idx] = "Mixed / transitional"
    return names


def estimate_cluster_stability(X: np.ndarray, n_clusters: int, random_state: int = 42, n_boot: int = 6) -> float:
    if len(X) < max(100, n_clusters * 10):
        return np.nan
    base_labels = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20).fit_predict(X)
    scores = []
    rng = np.random.default_rng(random_state)
    for i in range(n_boot):
        sample_idx = np.sort(rng.choice(len(X), size=int(len(X) * 0.8), replace=False))
        km = KMeans(n_clusters=n_clusters, random_state=random_state + i + 1, n_init=20)
        labels_sample = km.fit_predict(X[sample_idx])
        compare = adjusted_rand_score(base_labels[sample_idx], labels_sample)
        scores.append(compare)
    return float(np.mean(scores)) if scores else np.nan


def build_clusters(outcomes_df: pd.DataFrame, features: List[str], n_clusters: int = 6) -> Tuple[pd.DataFrame, ClusterArtifacts]:
    feat_df = outcomes_df[features].apply(pd.to_numeric, errors="coerce")
    valid = feat_df.dropna()
    if len(valid) < max(MIN_FEATURE_OBS, n_clusters * 10):
        raise ValueError("Not enough fully populated observations to build stable clusters.")
    valid_idx = valid.index
    n_clusters = max(3, min(n_clusters, max(3, len(valid) // 60)))
    scaler = StandardScaler()
    X = scaler.fit_transform(valid)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = km.fit_predict(X)
    cluster_stats_rows = []
    for cl in sorted(np.unique(labels)):
        sub = valid.loc[labels == cl]
        metrics = {
            "cluster": int(cl),
            "samples": int((labels == cl).sum()),
            "bounce_rate": float(outcomes_df.loc[sub.index, "bounce"].mean()),
            "repair_rate": float(outcomes_df.loc[sub.index, "repair"].mean()),
            "regime_rate": float(outcomes_df.loc[sub.index, "regime"].mean()),
        }
        for feat in features:
            metrics[f"{feat}_median"] = float(sub[feat].median())
        cluster_stats_rows.append(metrics)
    stats_df = pd.DataFrame(cluster_stats_rows).sort_values("cluster").reset_index(drop=True)
    names = assign_cluster_names(stats_df)
    stats_df["cluster_name"] = stats_df["cluster"].map(names)
    sil = float(silhouette_score(X, labels)) if len(np.unique(labels)) >= 2 else np.nan
    stability = estimate_cluster_stability(X, n_clusters)
    artifacts = ClusterArtifacts(
        scaler_mean=scaler.mean_.tolist(),
        scaler_scale=scaler.scale_.tolist(),
        features=features,
        centroids=km.cluster_centers_.tolist(),
        cluster_names={str(k): v for k, v in names.items()},
        cluster_stats=stats_df.set_index("cluster").to_dict(orient="index"),
        silhouette=sil,
        stability=stability,
    )
    return stats_df, artifacts


def predict_cluster(current: Dict[str, float], artifacts: ClusterArtifacts) -> Tuple[Optional[int], Optional[str], Optional[float]]:
    vals = []
    for feat in artifacts.features:
        v = safe_float(current.get(feat, np.nan))
        if pd.isna(v):
            return None, None, None
        vals.append(v)
    x = np.array(vals)
    scale = np.where(np.array(artifacts.scaler_scale) == 0, 1, np.array(artifacts.scaler_scale))
    scaled = (x - np.array(artifacts.scaler_mean)) / scale
    cents = np.array(artifacts.centroids)
    dists = np.sqrt(((cents - scaled) ** 2).sum(axis=1))
    cl = int(np.argmin(dists))
    confidence = 1.0 / (1.0 + float(dists[cl]))
    return cl, artifacts.cluster_names.get(str(cl), f"Cluster {cl}"), confidence

# -----------------------------
# Sweet spot scoring
# -----------------------------
def band_membership_score(value: float, median: float, q25: float, q75: float, std: float = 0.0) -> float:
    if pd.isna(value) or pd.isna(median) or pd.isna(q25) or pd.isna(q75):
        return np.nan
    if q25 <= value <= q75:
        return 1.0
    width = max(abs(q75 - q25), std, 1e-6)
    dist = abs(value - median)
    return float(max(0.0, min(1.0, 1.0 - (dist / (2.5 * width)))))


def score_snapshot_against_bands(snapshot: Dict[str, float], sweet_spots: Dict[str, Dict[str, Dict[str, float]]], weights: Optional[Dict[str, float]] = None) -> Tuple[pd.DataFrame, Dict[str, float]]:
    weights = weights or {feat: 1.0 for feat in KEY_FEATURES_FOR_SCORING}
    rows, totals = [], {}
    for label in ["bounce", "repair", "regime"]:
        label_bands = sweet_spots.get(label, {})
        weighted_sum = 0.0
        weight_total = 0.0
        for feat, meta in label_bands.items():
            if feat not in KEY_FEATURES_FOR_SCORING:
                continue
            cur = safe_float(snapshot.get(feat, np.nan))
            sc = band_membership_score(cur, meta["median"], meta["q25"], meta["q75"], meta.get("std", 0.0))
            w = weights.get(feat, 1.0)
            if pd.notna(sc):
                weighted_sum += w * sc
                weight_total += w
            rows.append({"Outcome": label.title(), "Feature": feat, "Current": cur, "Median": meta["median"], "Q25": meta["q25"], "Q75": meta["q75"], "Band Score": sc})
        totals[label] = 100.0 * weighted_sum / weight_total if weight_total else np.nan
    return pd.DataFrame(rows), totals


def weekly_overlay_score(snapshot_weekly: Dict[str, float], weekly_bands: Dict[str, Dict[str, float]]) -> Tuple[pd.DataFrame, float]:
    rows, vals = [], []
    for feat, meta in weekly_bands.items():
        cur = safe_float(snapshot_weekly.get(feat, np.nan))
        sc = band_membership_score(cur, meta["median"], meta["q25"], meta["q75"], meta.get("std", 0.0))
        rows.append({"Feature": feat, "Current": cur, "Median": meta["median"], "Q25": meta["q25"], "Q75": meta["q75"], "Score": sc})
        if pd.notna(sc):
            vals.append(sc)
    return pd.DataFrame(rows), (100.0 * float(np.mean(vals)) if vals else np.nan)

# -----------------------------
# Canary overlay
# -----------------------------
def build_canary_overlay_from_history(wide_daily: pd.DataFrame) -> pd.DataFrame:
    canary_rows = []
    for feat in CANARY_FEATURES:
        if feat not in wide_daily.columns:
            continue
        px = pd.to_numeric(wide_daily[feat], errors="coerce")
        if px.notna().sum() < 100:
            continue
        ma20 = px.rolling(20).mean()
        ma50 = px.rolling(50).mean()
        r = rsi(px, 14)
        t, _ = tsi(px, 25, 13, 7)
        score = ((px > ma20).astype(float) + (ma20 > ma50).astype(float) + (r > 50).astype(float) + (t > 0).astype(float)) / 4.0
        invert = feat == "SPXS:SVOL"
        if invert:
            score = 1.0 - score
        canary_rows.append(score.rename(feat))
    if not canary_rows:
        return pd.DataFrame(index=wide_daily.index)
    df = pd.concat(canary_rows, axis=1)
    df["canary_score"] = 100.0 * df.mean(axis=1, skipna=True)
    df["canary_confidence"] = 100.0 * df.notna().mean(axis=1)
    return df

# -----------------------------
# Proxy governance + actions
# -----------------------------
def compute_proxy_nymo(snapshot: Dict[str, float], prev_snapshot: Dict[str, float]) -> Dict[str, float]:
    nyad = safe_float(snapshot.get("$NYAD", np.nan))
    spxadp = safe_float(snapshot.get("$SPXADP", np.nan))
    prev_nyad = safe_float(prev_snapshot.get("$NYAD", np.nan))
    prev_spxadp = safe_float(prev_snapshot.get("$SPXADP", np.nan))
    cur_raw = 0.6 * (0 if pd.isna(nyad) else nyad) + 0.4 * (0 if pd.isna(spxadp) else spxadp)
    prev_raw = 0.6 * (0 if pd.isna(prev_nyad) else prev_nyad) + 0.4 * (0 if pd.isna(prev_spxadp) else prev_spxadp)
    proxy_nymo = 100 * np.tanh(cur_raw / 1600.0)
    prev_proxy_nymo = 100 * np.tanh(prev_raw / 1600.0)
    proxy_delta = proxy_nymo - prev_proxy_nymo
    state = "Unavailable"
    if not pd.isna(proxy_nymo):
        if proxy_nymo <= -70:
            state = "Deep washout"
        elif proxy_nymo <= -20:
            state = "Negative but repairing" if proxy_delta > 0 else "Negative and weak"
        elif proxy_nymo <= 20:
            state = "Neutral / crossing"
        else:
            state = "Positive thrust"
    return {"proxy_nymo": proxy_nymo, "proxy_delta": proxy_delta, "proxy_state": state}


def nymo_effective(snapshot: Dict[str, float], prev_snapshot: Dict[str, float], use_proxy: bool) -> Dict[str, float]:
    official = safe_float(snapshot.get("$NYMO", np.nan))
    prior = safe_float(prev_snapshot.get("$NYMO", np.nan))
    proxy = compute_proxy_nymo(snapshot, prev_snapshot)
    if use_proxy or pd.isna(official):
        return {"label": "NYMO Proxy", "value": proxy["proxy_nymo"], "delta": proxy["proxy_delta"], "mode": "Proxy", "state": proxy["proxy_state"]}
    return {"label": "Official NYMO", "value": official, "delta": official - prior if pd.notna(prior) else np.nan, "mode": "Official", "state": "Official series"}


def action_hierarchy(bounce_score: float, repair_score: float, regime_score: float, weekly_score: float,
                     cluster_name: Optional[str], nymo_eff: Dict[str, float], canary_score: float,
                     walkforward_alpha: float) -> Dict[str, object]:
    existing = "Stay defensive / monitor"
    new = "No new long"
    add = "Do not add"
    rsp_size = 0.0
    ursp_size = 0.0
    rationale = "Signals do not yet align strongly enough."

    canary_ok = pd.notna(canary_score) and canary_score >= 55
    wf_ok = pd.notna(walkforward_alpha) and walkforward_alpha >= 0
    if (repair_score >= 70 and regime_score >= 60 and weekly_score >= 60 and nymo_eff["value"] > -20 and
        cluster_name in {"Repair cluster", "Durable regime"} and canary_ok and wf_ok):
        existing = "Keep long bias"
        new = "New RSP okay; URSP selectively allowed"
        add = "Can add on confirmation holds"
        rsp_size = 0.30
        ursp_size = 0.10 if regime_score >= 72 and weekly_score >= 70 and canary_score >= 62 else 0.0
        rationale = "Empirical repair/regime sweet spots align, the canary overlay confirms, and walk-forward alpha is positive."
    elif bounce_score >= 65 and repair_score >= 55 and canary_ok:
        existing = "Hold / keep probe"
        new = "New RSP okay"
        add = "Add only after follow-through"
        rsp_size = 0.20
        rationale = "Current breadth matches historical bounce/repair zones reasonably well and the canaries are not contradicting it."
    elif bounce_score >= 55:
        existing = "Small probe only if already engaged"
        new = "New probe RSP only"
        add = "Do not add yet"
        rsp_size = 0.10
        rationale = "The setup resembles a bounce zone, but confirmation is incomplete."

    return {"existing": existing, "new": new, "add": add, "rsp_size": rsp_size, "ursp_size": ursp_size, "rationale": rationale}


def classify_delta(sym: str, cur: float, prev: float) -> str:
    if pd.isna(cur) or pd.isna(prev):
        return "n/a"
    d = cur - prev
    if sym == "$BPSPX_%B":
        if d >= 0.12: return "Shock+"
        if d >= 0.05: return "Thrust"
        if d <= -0.12: return "Shock-"
        if d <= -0.05: return "Collapse"
    elif sym in {"$SPXA50R", "$BPNYA", "$OEXA200R", "$BPSPX"}:
        if d >= 8: return "Shock+"
        if d >= 3: return "Thrust"
        if d <= -8: return "Shock-"
        if d <= -3: return "Collapse"
    elif sym in {"$NYMO", "$NYSI", "$NYHL", "$NYAD", "$SPXADP"}:
        if d >= 25: return "Shock+"
        if d >= 10: return "Thrust"
        if d <= -25: return "Shock-"
        if d <= -10: return "Collapse"
    elif sym == "$CPCE":
        if d >= 0.12: return "Fear spike"
        if d <= -0.12: return "Fear fade"
    elif sym == "RSP:SPY":
        if d >= 0.01: return "Leadership thrust"
        if d <= -0.01: return "Leadership fade"
    return "Improve" if d > 0 else "Fade" if d < 0 else "Flat"

# -----------------------------
# Backtests
# -----------------------------
def breadth_oscillator(score_series: pd.Series, fast: int = 5, slow: int = 13) -> pd.Series:
    ef = ema(score_series, fast)
    es = ema(score_series, slow)
    return (ef - es) / es.replace(0, np.nan)


def backtest_from_signal(df: pd.DataFrame, signal: pd.Series, asset_col: str = "RSP", switch_cost_bps: float = 0.0) -> pd.DataFrame:
    out = df[[asset_col]].copy().dropna()
    out["asset_ret"] = out[asset_col].pct_change().fillna(0)
    out["signal"] = signal.reindex(out.index).fillna(0).astype(float).shift(1).fillna(0)
    switch = out["signal"].diff().abs().fillna(0)
    cost = (switch_cost_bps / 10000.0) * switch
    out["strategy_ret"] = out["signal"] * out["asset_ret"] - cost
    out["equity_strategy"] = (1 + out["strategy_ret"]).cumprod()
    out["equity_buyhold"] = (1 + out["asset_ret"]).cumprod()
    return out.reset_index().rename(columns={"index": "date"})


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1
    return float(dd.min()) if len(dd) else np.nan


def sharpe_ratio(ret: pd.Series, periods_per_year: int = 252) -> float:
    s = pd.to_numeric(ret, errors="coerce").dropna()
    if len(s) < 2 or s.std() == 0:
        return np.nan
    return float(np.sqrt(periods_per_year) * s.mean() / s.std())


def walkforward_validation(outcomes: pd.DataFrame, sweet_spot_features: List[str], step: int = 20, min_train: int = 252) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows = []
    dates = outcomes.index.sort_values()
    for end_ix in range(min_train, len(dates) - 61, step):
        train = outcomes.iloc[:end_ix].copy()
        test = outcomes.iloc[end_ix:end_ix + step].copy()
        sweet_spots = {lbl: summarize_feature_bands(train, lbl, sweet_spot_features) for lbl in ["bounce", "repair", "regime"]}
        for dt, row in test.iterrows():
            _, totals = score_snapshot_against_bands(row.to_dict(), sweet_spots)
            mean_score = np.nanmean([totals.get("bounce", np.nan), totals.get("repair", np.nan), totals.get("regime", np.nan)])
            rows.append({
                "date": dt,
                "wf_score": mean_score,
                "ret_20": row.get("ret_20", np.nan),
                "ret_60": row.get("ret_60", np.nan),
                "regime": row.get("regime", False),
            })
    wf = pd.DataFrame(rows).sort_values("date") if rows else pd.DataFrame()
    if wf.empty:
        return wf, {"alpha": np.nan, "hit_rate": np.nan}
    hi = wf[wf["wf_score"] >= wf["wf_score"].quantile(0.70)]
    lo = wf[wf["wf_score"] <= wf["wf_score"].quantile(0.30)]
    alpha = hi["ret_20"].mean() - lo["ret_20"].mean() if not hi.empty and not lo.empty else np.nan
    hit = float((hi["ret_20"] > 0).mean()) if not hi.empty else np.nan
    return wf, {"alpha": alpha, "hit_rate": hit}

# -----------------------------
# Persistence
# -----------------------------
def save_model_artifacts(daily_feat, weekly_feat, sweet_spots, weekly_bands, combo_tables, cluster_stats, cluster_artifacts, outcomes, canary_df, wf_df, wf_stats, issues):
    daily_feat.to_parquet(HIST_DAILY_PATH, index=False)
    weekly_feat.to_parquet(HIST_WEEKLY_PATH, index=False)
    outcomes.to_parquet(APP_DIR / "outcomes.parquet")
    canary_df.to_parquet(APP_DIR / "canary.parquet") if not canary_df.empty else None
    wf_df.to_parquet(APP_DIR / "walkforward.parquet") if not wf_df.empty else None
    save_json(SWEET_SPOT_PATH, {"daily": sweet_spots, "weekly": weekly_bands, "combos": combo_tables, "issues": issues, "wf_stats": wf_stats})
    save_json(CLUSTER_INFO_PATH, {"cluster_stats": cluster_stats.to_dict(orient="records"), "artifacts": asdict(cluster_artifacts)})
    save_json(MODEL_META_PATH, {"saved_at": datetime.now().isoformat(), "daily_rows": int(len(daily_feat)), "weekly_rows": int(len(weekly_feat))})


def load_model_artifacts():
    if not HIST_DAILY_PATH.exists() or not SWEET_SPOT_PATH.exists() or not CLUSTER_INFO_PATH.exists():
        return None
    daily_feat = pd.read_parquet(HIST_DAILY_PATH)
    weekly_feat = pd.read_parquet(HIST_WEEKLY_PATH) if HIST_WEEKLY_PATH.exists() else pd.DataFrame()
    outcomes = pd.read_parquet(APP_DIR / "outcomes.parquet") if (APP_DIR / "outcomes.parquet").exists() else None
    canary_df = pd.read_parquet(APP_DIR / "canary.parquet") if (APP_DIR / "canary.parquet").exists() else pd.DataFrame()
    wf_df = pd.read_parquet(APP_DIR / "walkforward.parquet") if (APP_DIR / "walkforward.parquet").exists() else pd.DataFrame()
    sweet_json = load_json(SWEET_SPOT_PATH, {})
    cluster_json = load_json(CLUSTER_INFO_PATH, {})
    artifacts = ClusterArtifacts(**cluster_json.get("artifacts", {})) if cluster_json.get("artifacts") else None
    cluster_stats = pd.DataFrame(cluster_json.get("cluster_stats", []))
    return daily_feat, weekly_feat, sweet_json, cluster_stats, artifacts, outcomes, canary_df, wf_df


def append_upload_history(row: Dict):
    hist = pd.read_csv(UPLOAD_HISTORY_PATH) if UPLOAD_HISTORY_PATH.exists() else pd.DataFrame()
    hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    hist.to_csv(UPLOAD_HISTORY_PATH, index=False)

# -----------------------------
# Build model
# -----------------------------
def find_missing_core_features(wide_daily: pd.DataFrame) -> List[str]:
    return [f for f in ["RSP", *KEY_FEATURES_FOR_SCORING] if f not in wide_daily.columns]


def build_empirical_model(file_bytes: bytes, n_clusters: int = 6) -> Dict[str, object]:
    daily, weekly, issues = parse_stockcharts_zip(file_bytes)
    daily_feat = add_indicator_features(daily)
    weekly_feat = add_indicator_features(weekly) if not weekly.empty else pd.DataFrame()
    daily_wide = wide_from_hist(daily_feat)
    weekly_wide = wide_from_hist(weekly_feat) if not weekly_feat.empty else pd.DataFrame()
    missing = find_missing_core_features(daily_wide)
    if "RSP" in missing:
        raise ValueError("Historical zip must include RSP daily history.")
    outcomes = label_outcomes(daily_wide)
    sweet_spots = {label: summarize_feature_bands(outcomes, label, KEY_FEATURES_FOR_SCORING) for label in ["bounce", "repair", "regime"]}
    combo_tables = {label: best_combo_zones(outcomes, label, KEY_FEATURES_FOR_SCORING).to_dict(orient="records") for label in ["bounce", "repair", "regime"]}
    weekly_bands = {}
    if not weekly_wide.empty:
        for feat in WEEKLY_FEATURES:
            if feat not in weekly_wide.columns:
                continue
            s = pd.to_numeric(weekly_wide[feat], errors="coerce").dropna()
            if len(s) < 10:
                continue
            weekly_bands[feat] = {"median": float(s.median()), "q25": float(s.quantile(0.25)), "q75": float(s.quantile(0.75)), "std": float(0 if pd.isna(s.std()) else s.std())}
    cluster_base = outcomes.dropna(subset=[f for f in KEY_FEATURES_FOR_SCORING if f in outcomes.columns]).copy()
    cluster_features = [f for f in KEY_FEATURES_FOR_SCORING if f in cluster_base.columns]
    cluster_stats, cluster_artifacts = build_clusters(cluster_base, cluster_features, n_clusters=n_clusters)
    canary_df = build_canary_overlay_from_history(daily_wide)
    wf_df, wf_stats = walkforward_validation(outcomes, [f for f in KEY_FEATURES_FOR_SCORING if f in outcomes.columns])
    issues.extend([f"Missing daily features: {', '.join(missing)}"] if missing else [])
    save_model_artifacts(daily_feat, weekly_feat, sweet_spots, weekly_bands, combo_tables, cluster_stats, cluster_artifacts, outcomes, canary_df, wf_df, wf_stats, issues)
    return {
        "daily_feat": daily_feat,
        "weekly_feat": weekly_feat,
        "sweet_spots": sweet_spots,
        "weekly_bands": weekly_bands,
        "cluster_stats": cluster_stats,
        "cluster_artifacts": cluster_artifacts,
        "combo_tables": combo_tables,
        "outcomes": outcomes,
        "canary_df": canary_df,
        "wf_df": wf_df,
        "wf_stats": wf_stats,
        "issues": issues,
        "missing": missing,
    }

# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("Model Inputs")
phase = market_phase()
st.sidebar.markdown(f"**Market phase:** {phase}")
auto_use_proxy = phase in {"Morning", "Midday", "Power Hour", "Post Close"}
use_proxy = st.sidebar.toggle("Use proxy NYMO before official evening data", value=auto_use_proxy, help="Proxy NYMO uses (NYAD×0.6) + (SPXADP×0.4), compressed into a NYMO-like range.")
show_debug = st.sidebar.toggle("Show debug tables", value=False)
rebuild = st.sidebar.toggle("Force rebuild historical model", value=False)
n_clusters = st.sidebar.slider("Cluster count", 3, 8, 6)
switch_cost = st.sidebar.slider("Backtest switch cost (bps)", 0.0, 25.0, 2.0, 0.5)

st.sidebar.markdown("---")
hist_upload = st.sidebar.file_uploader("One-time historical upload (.zip)", type=["zip"])
rt_upload = st.sidebar.file_uploader("Realtime snapshot upload (.csv)", type=["csv"])

# -----------------------------
# Load or build model with error handling
# -----------------------------
model = None
model_errors = []
if hist_upload is not None and (rebuild or not HIST_DAILY_PATH.exists()):
    try:
        with st.spinner("Building empirical sweet-spot + clustering model from historical zip..."):
            model = build_empirical_model(hist_upload.read(), n_clusters=n_clusters)
            st.sidebar.success("Historical model rebuilt.")
    except Exception as e:
        model_errors.append(str(e))

if model is None:
    loaded = load_model_artifacts()
    if loaded is not None:
        daily_feat, weekly_feat, sweet_json, cluster_stats, cluster_artifacts, outcomes, canary_df, wf_df = loaded
        model = {
            "daily_feat": daily_feat,
            "weekly_feat": weekly_feat,
            "sweet_spots": sweet_json.get("daily", {}),
            "weekly_bands": sweet_json.get("weekly", {}),
            "combo_tables": sweet_json.get("combos", {}),
            "cluster_stats": cluster_stats,
            "cluster_artifacts": cluster_artifacts,
            "outcomes": outcomes if outcomes is not None else label_outcomes(wide_from_hist(daily_feat)),
            "canary_df": canary_df,
            "wf_df": wf_df,
            "wf_stats": sweet_json.get("wf_stats", {}),
            "issues": sweet_json.get("issues", []),
            "missing": [],
        }

if model is None:
    warning_box(model_errors, title="Build errors")
    st.info("Upload your one-time `stockcharts.zip` baseline in the sidebar to build the model.")
    st.stop()

warning_box(model_errors + model.get("issues", []), title="Model notes")

# -----------------------------
# Current snapshot
# -----------------------------
rt_df = None
snap_path = None
if rt_upload is not None:
    try:
        rt_df = parse_realtime_snapshot(rt_upload.read())
        snap_path = SNAPSHOT_DIR / f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        rt_df.to_csv(snap_path, index=False)
    except Exception as e:
        st.error(f"Realtime snapshot could not be parsed: {e}")

latest_snap, prev_snap, latest_date = current_snapshot_from_hist(model["daily_feat"], rt_df)
weekly_snap, weekly_prev, weekly_latest_date = current_snapshot_from_hist(model["weekly_feat"], None) if not model["weekly_feat"].empty else ({}, {}, None)

nymo_eff = nymo_effective(latest_snap, prev_snap, use_proxy)
band_df, total_scores = score_snapshot_against_bands(latest_snap, model["sweet_spots"])
weekly_df, weekly_score = weekly_overlay_score(weekly_snap, model["weekly_bands"]) if model["weekly_bands"] else (pd.DataFrame(), np.nan)
cluster_id, cluster_name, cluster_conf = predict_cluster(latest_snap, model["cluster_artifacts"]) if model["cluster_artifacts"] else (None, None, None)
canary_score = np.nan
if not model["canary_df"].empty:
    canary_score = safe_float(model["canary_df"]["canary_score"].iloc[-1])
walkforward_alpha = safe_float(model.get("wf_stats", {}).get("alpha", np.nan))
actions = action_hierarchy(total_scores.get("bounce", np.nan), total_scores.get("repair", np.nan), total_scores.get("regime", np.nan), weekly_score, cluster_name, nymo_eff, canary_score, walkforward_alpha)

append_upload_history({
    "upload_ts": datetime.now().isoformat(timespec="seconds"),
    "latest_hist_date": str(latest_date.date()) if pd.notna(latest_date) else None,
    "phase": phase,
    "use_proxy": use_proxy,
    "bounce_score": total_scores.get("bounce", np.nan),
    "repair_score": total_scores.get("repair", np.nan),
    "regime_score": total_scores.get("regime", np.nan),
    "weekly_score": weekly_score,
    "cluster": cluster_name,
    "canary_score": canary_score,
    "snapshot_file": str(snap_path) if snap_path else None,
})

# Alerts
alerts = []
if total_scores.get("repair", 0) >= 70 and canary_score >= 55:
    alerts.append("Repair score and canary overlay are aligned.")
if cluster_name == "Capitulation / washout" and total_scores.get("bounce", 0) >= 60:
    alerts.append("Washout cluster plus strong bounce score: probe conditions may be forming.")
if pd.notna(weekly_score) and weekly_score < 40:
    alerts.append("Weekly overlay is weak. Be careful adding leverage.")
if nymo_eff["mode"] == "Proxy":
    alerts.append("Using proxy NYMO. Re-check after official EOD breadth refresh.")

# -----------------------------
# Top layout
# -----------------------------
col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.markdown("<div style='font-size:1.05rem;font-weight:800;'>Realtime decision frame</div>", unsafe_allow_html=True)
    pills = [f"<span class='pill pill-blue'>{phase}</span>", f"<span class='pill {'pill-yellow' if use_proxy else 'pill-green'}'>{nymo_eff['label']}</span>"]
    if cluster_name:
        pills.append(f"<span class='pill pill-purple'>{cluster_name}</span>")
    st.markdown("".join(pills), unsafe_allow_html=True)
    st.markdown(
        f"<div class='kpi-grid'>"
        f"<div class='kpi-box'><div class='kpi-label'>Latest baseline date</div><div class='kpi-value'>{latest_date.date() if pd.notna(latest_date) else 'n/a'}</div></div>"
        f"<div class='kpi-box'><div class='kpi-label'>Cluster confidence</div><div class='kpi-value'>{'n/a' if cluster_conf is None else f'{cluster_conf*100:.0f}%'}</div></div>"
        f"<div class='kpi-box'><div class='kpi-label'>Canary score</div><div class='kpi-value'>{fmt_num(canary_score,0)}</div></div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
with col2:
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.markdown("<div style='font-size:1.05rem;font-weight:800;'>Action hierarchy</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='action-box action-existing'><b>Existing:</b> {actions['existing']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='action-box action-new'><b>New:</b> {actions['new']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='action-box action-add'><b>Add:</b> {actions['add']}</div>", unsafe_allow_html=True)
    st.caption(f"Target sizing → RSP: {actions['rsp_size']:.0%} | URSP: {actions['ursp_size']:.0%}. {actions['rationale']}")
    st.markdown("</div>", unsafe_allow_html=True)

r1, r2, r3, r4 = st.columns(4)
with r1: score_card("Bounce score", float(total_scores.get("bounce", np.nan) or 0), 100, "Historical bounce sweet-spot match")
with r2: score_card("Repair score", float(total_scores.get("repair", np.nan) or 0), 100, "Historical repair sweet-spot match")
with r3: score_card("Regime score", float(total_scores.get("regime", np.nan) or 0), 100, "Durable participation regime match")
with r4: score_card("Weekly overlay", float(weekly_score if not pd.isna(weekly_score) else 0), 100, "Weekly backdrop / durability filter")

if alerts:
    st.info("\n".join([f"• {x}" for x in alerts]))

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Decision Dashboard", "Oscillator + Backtest", "Sweet Spot Explorer", "Validation + Export"])

with tab1:
    c1, c2 = st.columns([1.1, 0.9])
    with c1:
        st.subheader("Sweet-spot scoring detail")
        show_cols = ["Outcome", "Feature", "Current", "Median", "Q25", "Q75", "Band Score"]
        st.dataframe(band_df[show_cols].round(3), use_container_width=True, hide_index=True)
        st.subheader("Momentum context")
        momentum_features = [f for f in ["$BPSPX_%B", "$BPNYA", "$OEXA200R", "$SPXA50R", "$NYMO", "$NYSI", "$CPCE", "$NYHL", "RSP:SPY"] if f in latest_snap or f in prev_snap]
        m_rows = []
        for sym in momentum_features:
            cur = safe_float(latest_snap.get(sym, np.nan))
            prev = safe_float(prev_snap.get(sym, np.nan))
            m_rows.append({"Feature": sym, "Current": cur, "Prior": prev, "Delta": cur - prev if pd.notna(cur) and pd.notna(prev) else np.nan, "Class": classify_delta(sym, cur, prev)})
        st.dataframe(pd.DataFrame(m_rows).round(3), use_container_width=True, hide_index=True)
    with c2:
        st.subheader("Proxy governance")
        st.markdown(f"**Mode:** {nymo_eff['mode']}")
        st.markdown(f"**{nymo_eff['label']}:** {fmt_num(nymo_eff['value'])}")
        st.markdown(f"**Delta:** {fmt_num(nymo_eff['delta'])}")
        st.markdown(f"**State:** {nymo_eff['state']}")
        st.subheader("Cluster regime")
        if cluster_name:
            st.markdown(f"**Current cluster:** {cluster_name}")
            st.markdown(f"**Confidence:** {cluster_conf*100:.1f}%")
            st.markdown(f"**Silhouette:** {fmt_num(model['cluster_artifacts'].silhouette,3)}")
            st.markdown(f"**Stability:** {fmt_num(model['cluster_artifacts'].stability,3)}")
            stats_map = model["cluster_artifacts"].cluster_stats.get(str(cluster_id), {}) if model["cluster_artifacts"] else {}
            if stats_map:
                st.markdown(f"Bounce hit rate: {stats_map.get('bounce_rate', np.nan):.1%}")
                st.markdown(f"Repair hit rate: {stats_map.get('repair_rate', np.nan):.1%}")
                st.markdown(f"Regime hit rate: {stats_map.get('regime_rate', np.nan):.1%}")
        if not weekly_df.empty:
            st.subheader("Weekly overlay detail")
            st.dataframe(weekly_df.round(3), use_container_width=True, hide_index=True)
        if not model["canary_df"].empty:
            st.subheader("Canary overlay")
            st.markdown(f"**Latest canary score:** {fmt_num(canary_score,1)}")
            st.line_chart(model["canary_df"][["canary_score"]].tail(180), height=180)

    st.subheader("Breadth chart panel")
    chart_syms = [s for s in ["$BPSPX", "$SPXA50R", "$NYMO", "$NYSI", "$CPCE", "RSP"] if s in model["daily_feat"]["symbol"].unique()]
    sel = st.selectbox("Chart symbol", options=chart_syms, index=0 if chart_syms else None)
    if sel:
        g = model["daily_feat"][model["daily_feat"]["symbol"] == sel].sort_values("date").tail(180).copy()
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.55, 0.22, 0.23])
        fig.add_trace(go.Candlestick(x=g["date"], open=g["open"], high=g["high"], low=g["low"], close=g["close"], name=sel), row=1, col=1)
        fig.add_trace(go.Scatter(x=g["date"], y=g["ma20"], name="MA20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=g["date"], y=g["ma50"], name="MA50"), row=1, col=1)
        fig.add_trace(go.Scatter(x=g["date"], y=g["pct_b20"], name="%B"), row=2, col=1)
        fig.add_trace(go.Scatter(x=g["date"], y=g["rsi14"], name="RSI14"), row=2, col=1)
        fig.add_trace(go.Scatter(x=g["date"], y=g["cci20"], name="CCI20"), row=3, col=1)
        fig.update_layout(height=780, xaxis_rangeslider_visible=False, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Breadth oscillator backtest")
    outcomes = model["outcomes"].copy().sort_index()
    score_rows = []
    for dt, row in outcomes.iterrows():
        _, totals = score_snapshot_against_bands(row.to_dict(), model["sweet_spots"])
        score_rows.append(np.nanmean([totals.get("bounce", np.nan), totals.get("repair", np.nan), totals.get("regime", np.nan)]))
    outcomes["sweetspot_score"] = score_rows
    outcomes["oscillator"] = breadth_oscillator(outcomes["sweetspot_score"])
    signal = (outcomes["oscillator"] > 0).astype(float)
    if not model["canary_df"].empty:
        signal = signal * ((model["canary_df"]["canary_score"].reindex(outcomes.index).fillna(0) >= 50).astype(float))
    bt = backtest_from_signal(outcomes, signal, asset_col="RSP", switch_cost_bps=switch_cost)
    if not bt.empty:
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Strategy return", f"{(bt['equity_strategy'].iloc[-1]-1):.1%}")
        with c2: st.metric("Buy & hold", f"{(bt['equity_buyhold'].iloc[-1]-1):.1%}")
        with c3: st.metric("Strategy max DD", f"{max_drawdown(bt['equity_strategy']):.1%}")
        with c4: st.metric("Strategy Sharpe", f"{fmt_num(sharpe_ratio(bt['strategy_ret']),2)}")
        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.62, 0.38])
        fig2.add_trace(go.Scatter(x=bt["date"], y=bt["equity_strategy"], name="Strategy", line=dict(width=2)), row=1, col=1)
        fig2.add_trace(go.Scatter(x=bt["date"], y=bt["equity_buyhold"], name="Buy & Hold", line=dict(width=2)), row=1, col=1)
        fig2.add_trace(go.Scatter(x=outcomes.index, y=outcomes["oscillator"], name="Breadth Oscillator", line=dict(width=2)), row=2, col=1)
        fig2.add_hline(y=0, row=2, col=1)
        fig2.update_layout(height=680, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Backtest unavailable; insufficient historical alignment.")

with tab3:
    st.subheader("Empirical sweet-spot explorer")
    ss_choice = st.selectbox("Outcome", ["bounce", "repair", "regime"], index=1)
    ss_table = pd.DataFrame(model["sweet_spots"].get(ss_choice, {})).T.reset_index().rename(columns={"index": "Feature"})
    if not ss_table.empty:
        st.dataframe(ss_table.round(3), use_container_width=True, hide_index=True)
    combos = pd.DataFrame(model["combo_tables"].get(ss_choice, []))
    if not combos.empty:
        st.subheader("Top combo zones")
        st.dataframe(combos.round(3), use_container_width=True, hide_index=True)
    if not model["cluster_stats"].empty:
        st.subheader("Cluster map")
        st.dataframe(model["cluster_stats"].round(3), use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Validation + export")
    wf_df = model.get("wf_df", pd.DataFrame())
    wf_stats = model.get("wf_stats", {})
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Walk-forward alpha (top vs bottom score bucket, 20d fwd return):** {fmt_num(wf_stats.get('alpha', np.nan), 3)}")
        st.markdown(f"**Walk-forward positive hit rate (top score bucket):** {fmt_num(100*safe_float(wf_stats.get('hit_rate', np.nan)), 1)}%")
        if model["cluster_artifacts"]:
            st.markdown(f"**Cluster silhouette:** {fmt_num(model['cluster_artifacts'].silhouette, 3)}")
            st.markdown(f"**Cluster stability:** {fmt_num(model['cluster_artifacts'].stability, 3)}")
    with c2:
        st.markdown("**Available exports**")
        st.download_button("Download sweet spots JSON", data=json.dumps(model["sweet_spots"], indent=2), file_name="sweet_spots.json", mime="application/json")
        st.download_button("Download cluster stats CSV", data=model["cluster_stats"].to_csv(index=False), file_name="cluster_stats.csv", mime="text/csv")
        if not wf_df.empty:
            st.download_button("Download walk-forward CSV", data=wf_df.to_csv(index=False), file_name="walkforward.csv", mime="text/csv")
    if not wf_df.empty:
        st.line_chart(wf_df.set_index("date")[["wf_score", "ret_20"]].tail(250), height=260)

if show_debug:
    st.markdown("---")
    st.write("Latest snapshot", latest_snap)
    st.write("Previous snapshot", prev_snap)
    st.write("Weekly snapshot", weekly_snap)
