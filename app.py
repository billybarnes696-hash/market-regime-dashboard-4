
import io
import json
import math
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =============================
# Config / constants
# =============================
st.set_page_config(page_title="Breadth Historical Gate Engine Pro", layout="wide", page_icon="📈")

APP_DIR = Path("gate_model_store")
APP_DIR.mkdir(exist_ok=True)
MODEL_JSON = APP_DIR / "learned_gate_model.json"
DAILY_HIST_PARQUET = APP_DIR / "daily_history.parquet"
WEEKLY_HIST_PARQUET = APP_DIR / "weekly_history.parquet"
UPLOAD_HISTORY_CSV = APP_DIR / "snapshot_upload_history.csv"

PRIMARY_FEATURES = [
    "$BPSPX", "$BPNYA", "$SPXA50R", "$OEXA50R", "$OEXA150R", "$OEXA200R",
    "$NYMO", "$NYSI", "$NYAD", "$SPXADP", "$NYHL", "$TRIN", "$CPCE", "$VIX", "VXX",
    "RSP", "SPY", "URSP", "RSP_SPY", "SMH_SPY", "IWM_SPY", "XLF_SPY", "HYG_IEF", "SPXS_SVOL"
]
TREND_FEATURE_BASES = [
    "$BPSPX", "$BPNYA", "$SPXA50R", "$OEXA200R", "$NYMO", "$NYSI", "$NYHL", "$TRIN", "$CPCE", "$VIX", "VXX",
    "RSP_SPY", "SMH_SPY", "IWM_SPY", "XLF_SPY", "HYG_IEF", "SPXS_SVOL"
]
TREND_WINDOWS = [1, 2, 3, 5, 10]
MIN_GATE_SUPPORT = 40
TOP_SINGLE_PER_OUTCOME = 10
MAX_COMBO_CANDIDATES = 8

OUTCOME_SPECS = {
    "bounce": {"max_days": 10, "good_up": 0.03, "bad_down": -0.03},
    "repair": {"max_days": 20, "good_up": 0.04, "bad_down": -0.05},
    "regime": {"max_days": 60, "good_up": 0.08, "bad_down": -0.08, "confirm_days": 20, "confirm_up": 0.03},
    "fall": {"max_days": 10, "good_down": -0.03, "bad_up": 0.03},
}

SYMBOL_MAP = {
    "_BPSPX": "$BPSPX", "_BPNYA": "$BPNYA", "_SPXA50R": "$SPXA50R", "_OEXA50R": "$OEXA50R",
    "_OEXA150R": "$OEXA150R", "_OEXA200R": "$OEXA200R", "_NYMO": "$NYMO", "_NYSI": "$NYSI",
    "_NYAD": "$NYAD", "_SPXADP": "$SPXADP", "_NYHL": "$NYHL", "_TRIN": "$TRIN", "_CPCE": "$CPCE",
    "_VIX": "$VIX", "_SPX": "$SPX", "RSP": "RSP", "SPY": "SPY", "URSP": "URSP", "VXX": "VXX",
    "RSP_SPY": "RSP_SPY", "SMH_SPY": "SMH_SPY", "IWM_SPY": "IWM_SPY", "XLF_SPY": "XLF_SPY",
    "HYG_IEF": "HYG_IEF", "SPXS_SVOL": "SPXS_SVOL"
}

CUSTOM_CSS = """
<style>
.block-container {padding-top:1rem; padding-bottom:2rem;}
.card {background: linear-gradient(180deg,#111a36,#0b1227); border:1px solid rgba(148,163,184,.22); border-radius:16px; padding:1rem; margin-bottom:1rem;}
.metric {background: rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.08); border-radius:14px; padding:.9rem;}
.signal-long {background:rgba(34,197,94,.14); border:1px solid rgba(34,197,94,.32); border-radius:16px; padding:1rem;}
.signal-short {background:rgba(239,68,68,.14); border:1px solid rgba(239,68,68,.32); border-radius:16px; padding:1rem;}
.signal-hold {background:rgba(245,158,11,.14); border:1px solid rgba(245,158,11,.32); border-radius:16px; padding:1rem;}
.smallmuted {color:#94a3b8; font-size:.9rem;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =============================
# Utility functions
# =============================
def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def pct_b(s: pd.Series, window: int = 20, stds: float = 2.0) -> pd.Series:
    ma = s.rolling(window).mean()
    sd = s.rolling(window).std()
    upper = ma + stds * sd
    lower = ma - stds * sd
    return (s - lower) / (upper - lower).replace(0, np.nan)

def rsi(s: pd.Series, period: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    ma_up = up.ewm(alpha=1 / period, adjust=False).mean()
    ma_dn = dn.ewm(alpha=1 / period, adjust=False).mean()
    rs = ma_up / ma_dn.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)

def cci_from_close(s: pd.Series, period: int = 20) -> pd.Series:
    sma = s.rolling(period).mean()
    mad = s.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (s - sma) / (0.015 * mad.replace(0, np.nan))

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def compute_proxy_nymo(nyad: pd.Series, spxadp: pd.Series) -> pd.Series:
    raw = 0.6 * nyad.fillna(0) + 0.4 * spxadp.fillna(0)
    return 100.0 * np.tanh(raw / 1600.0)

def compute_proxy_nysi(proxy_nymo: pd.Series) -> pd.Series:
    # slow cumulative breadth proxy
    return ema(proxy_nymo.fillna(0), 5).rolling(10, min_periods=1).sum()

def normalize_name(name: str) -> Tuple[str, str]:
    stem = Path(name).stem.strip()
    timeframe = "weekly" if stem.lower().endswith(" w") else "daily"
    stem = stem[:-2].strip() if timeframe == "weekly" else stem
    key = stem.replace(" ", "").replace("-", "_")
    if key.startswith("_"):
        key = key.upper()
    else:
        key = key.upper()
    return SYMBOL_MAP.get(key, key), timeframe

def append_upload_history(row: Dict):
    if UPLOAD_HISTORY_CSV.exists():
        hist = pd.read_csv(UPLOAD_HISTORY_CSV)
    else:
        hist = pd.DataFrame()
    hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    hist.to_csv(UPLOAD_HISTORY_CSV, index=False)

# =============================
# Parsing
# =============================
def parse_stockcharts_csv(content: bytes, member_name: str) -> pd.DataFrame:
    text = content.decode("utf-8", errors="ignore").splitlines()
    if len(text) < 3:
        raise ValueError(f"{member_name}: not enough lines")
    symbol, timeframe = normalize_name(member_name)
    rows = []
    for line in text[2:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        dt = pd.to_datetime(parts[0], errors="coerce")
        if pd.isna(dt):
            continue
        rows.append({
            "date": dt,
            "symbol": symbol,
            "open": safe_float(parts[1]),
            "high": safe_float(parts[2]),
            "low": safe_float(parts[3]),
            "close": safe_float(parts[4]),
            "volume": safe_float(parts[5]),
            "timeframe": timeframe,
        })
    return pd.DataFrame(rows)

def parse_stockcharts_zip(file_bytes: bytes) -> Tuple[pd.DataFrame, pd.DataFrame]:
    daily_frames, weekly_frames = [], []
    with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".csv"):
                continue
            try:
                df = parse_stockcharts_csv(zf.read(name), name)
            except Exception:
                continue
            if df.empty:
                continue
            if df["timeframe"].iloc[0] == "weekly":
                weekly_frames.append(df)
            else:
                daily_frames.append(df)
    if not daily_frames:
        raise ValueError("No daily CSV files were parsed from the ZIP.")
    daily = pd.concat(daily_frames, ignore_index=True).sort_values(["symbol", "date"])
    weekly = pd.concat(weekly_frames, ignore_index=True).sort_values(["symbol", "date"]) if weekly_frames else pd.DataFrame()
    return daily, weekly

def parse_snapshot_csv(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    if "Symbol" not in df.columns:
        raise ValueError("Snapshot CSV must contain a 'Symbol' column.")
    df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
    close_col = "Close" if "Close" in df.columns else next((c for c in df.columns if "close" in c.lower()), None)
    if close_col is None:
        raise ValueError("Snapshot CSV must contain a close-like column.")
    pct_col = next((c for c in df.columns if "pctchange" in c.lower() or "change" in c.lower()), None)
    out = pd.DataFrame({
        "symbol": df["Symbol"],
        "close": pd.to_numeric(df[close_col], errors="coerce"),
        "pct_change": pd.to_numeric(df[pct_col], errors="coerce") if pct_col else np.nan,
    })
    return out.dropna(subset=["close"])

# =============================
# Feature engineering
# =============================
def pivot_close(hist: pd.DataFrame) -> pd.DataFrame:
    return hist.pivot(index="date", columns="symbol", values="close").sort_index()

def build_level_and_trend_features(close_df: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame(index=close_df.index)
    for sym in close_df.columns:
        s = close_df[sym].astype(float)
        feats[sym] = s
        feats[f"{sym}__BBP"] = pct_b(s)
        feats[f"{sym}__RSI14"] = rsi(s, 14)
        feats[f"{sym}__CCI20"] = cci_from_close(s, 20)
        for w in TREND_WINDOWS:
            feats[f"{sym}__D{w}"] = s.diff(w)
            feats[f"{sym}__ROC{w}"] = 100.0 * (s / s.shift(w) - 1.0)
        feats[f"{sym}__BBP_D3"] = feats[f"{sym}__BBP"].diff(3)
        feats[f"{sym}__RSI14_D3"] = feats[f"{sym}__RSI14"].diff(3)
        feats[f"{sym}__CCI20_D3"] = feats[f"{sym}__CCI20"].diff(3)
    if "$NYAD" in close_df.columns and "$SPXADP" in close_df.columns:
        feats["NYMO_PROXY"] = compute_proxy_nymo(close_df["$NYAD"], close_df["$SPXADP"])
        feats["NYSI_PROXY"] = compute_proxy_nysi(feats["NYMO_PROXY"])
        for w in TREND_WINDOWS:
            feats[f"NYMO_PROXY__D{w}"] = feats["NYMO_PROXY"].diff(w)
            feats[f"NYSI_PROXY__D{w}"] = feats["NYSI_PROXY"].diff(w)
    return feats

def add_forward_outcomes(close_df: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    rsp = close_df["RSP"].dropna().copy()
    aligned = feats.loc[rsp.index].copy()
    for name, spec in OUTCOME_SPECS.items():
        h = spec["max_days"]
        fwd = pd.concat({i: rsp.shift(-i) / rsp - 1.0 for i in range(1, h + 1)}, axis=1)
        fwd_max = fwd.max(axis=1)
        fwd_min = fwd.min(axis=1)
        if name == "bounce":
            aligned[name] = (fwd_max >= spec["good_up"]) & (fwd_min >= spec["bad_down"])
        elif name == "repair":
            aligned[name] = (fwd.iloc[:, -1] >= spec["good_up"]) & (fwd_min >= spec["bad_down"])
        elif name == "regime":
            confirm = rsp.shift(-spec["confirm_days"]) / rsp - 1.0
            aligned[name] = (fwd.iloc[:, -1] >= spec["good_up"]) & (confirm >= spec["confirm_up"]) & (fwd_min >= spec["bad_down"])
        elif name == "fall":
            aligned[name] = (fwd_min <= spec["good_down"]) & (fwd_max <= spec["bad_up"])
    return aligned

def build_weekly_overlay(weekly_hist: pd.DataFrame) -> Dict:
    if weekly_hist.empty:
        return {"available": False}
    wclose = pivot_close(weekly_hist)
    wfeat = build_level_and_trend_features(wclose)
    latest = wfeat.iloc[-1].dropna()
    checks = {}
    for key in ["$NYSI", "$OEXA200R", "$SPXA50R", "$NYHL", "$BPSPX", "$BPNYA"]:
        if key in latest.index:
            if key in ["$NYSI", "$NYHL"]:
                checks[key] = latest[key] > latest.get(f"{key}__D3", np.nan)
            else:
                checks[key] = latest[key] > wfeat[key].quantile(0.45)
    pass_fraction = float(np.mean(list(checks.values()))) if checks else np.nan
    return {
        "available": True,
        "pass_fraction": pass_fraction,
        "checks": checks,
        "latest_date": str(wfeat.index.max().date()),
    }

# =============================
# Gate learning
# =============================
@dataclass
class Gate:
    outcome: str
    kind: str
    description: str
    conditions: List[Dict]
    support: int
    base_rate: float
    hit_rate: float
    lift: float
    score: float

def eval_condition(series: pd.Series, cond: Dict) -> pd.Series:
    op = cond["op"]
    thr = cond["thr"]
    if op == ">=":
        return series >= thr
    return series <= thr

def score_gate(pass_mask: pd.Series, outcome: pd.Series) -> Optional[Tuple[int, float, float, float]]:
    valid = pass_mask.notna() & outcome.notna()
    if valid.sum() < MIN_GATE_SUPPORT:
        return None
    passed = pass_mask[valid]
    support = int(passed.sum())
    if support < MIN_GATE_SUPPORT:
        return None
    hit_rate = float(outcome[valid & passed].mean())
    base_rate = float(outcome[valid].mean())
    if math.isnan(hit_rate) or math.isnan(base_rate) or base_rate <= 0:
        return None
    lift = hit_rate / base_rate
    return support, base_rate, hit_rate, lift

def choose_candidate_thresholds(series: pd.Series) -> List[float]:
    vals = series.dropna()
    if vals.nunique() < 12:
        return sorted(vals.unique().tolist())
    qs = np.linspace(0.15, 0.85, 8)
    return sorted(set(float(vals.quantile(q)) for q in qs))

def learn_single_feature_gates(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, List[Gate]]:
    out = {k: [] for k in OUTCOME_SPECS.keys()}
    for outcome in OUTCOME_SPECS.keys():
        y = df[outcome].astype(bool)
        for feat in feature_cols:
            s = pd.to_numeric(df[feat], errors="coerce")
            if s.notna().sum() < MIN_GATE_SUPPORT * 2:
                continue
            thresholds = choose_candidate_thresholds(s)
            best: Optional[Gate] = None
            for thr in thresholds:
                for op in (">=", "<="):
                    pmask = eval_condition(s, {"op": op, "thr": thr})
                    scored = score_gate(pmask, y)
                    if not scored:
                        continue
                    support, base_rate, hit_rate, lift = scored
                    # prefer higher lift, but require decent hit rate lift and not vanishing support
                    score = (lift - 1.0) * np.sqrt(support)
                    gate = Gate(
                        outcome=outcome,
                        kind="single",
                        description=f"{feat} {op} {thr:.3f}",
                        conditions=[{"feature": feat, "op": op, "thr": float(thr)}],
                        support=support,
                        base_rate=base_rate,
                        hit_rate=hit_rate,
                        lift=lift,
                        score=score,
                    )
                    if (best is None) or (gate.score > best.score):
                        best = gate
            if best:
                out[outcome].append(best)
        out[outcome] = sorted(out[outcome], key=lambda g: g.score, reverse=True)[:TOP_SINGLE_PER_OUTCOME]
    return out

def learn_combo_gates(df: pd.DataFrame, single_gates: Dict[str, List[Gate]]) -> Dict[str, List[Gate]]:
    combo_out = {k: [] for k in OUTCOME_SPECS.keys()}
    for outcome, gates in single_gates.items():
        y = df[outcome].astype(bool)
        cand = gates[:MAX_COMBO_CANDIDATES]
        best_combos = []
        for i in range(len(cand)):
            for j in range(i + 1, len(cand)):
                g1, g2 = cand[i], cand[j]
                c1, c2 = g1.conditions[0], g2.conditions[0]
                if c1["feature"] == c2["feature"]:
                    continue
                s1 = pd.to_numeric(df[c1["feature"]], errors="coerce")
                s2 = pd.to_numeric(df[c2["feature"]], errors="coerce")
                pmask = eval_condition(s1, c1) & eval_condition(s2, c2)
                scored = score_gate(pmask, y)
                if not scored:
                    continue
                support, base_rate, hit_rate, lift = scored
                score = (lift - 1.0) * np.sqrt(support)
                best_combos.append(Gate(
                    outcome=outcome,
                    kind="combo",
                    description=f"{g1.description} AND {g2.description}",
                    conditions=[c1, c2],
                    support=support, base_rate=base_rate, hit_rate=hit_rate, lift=lift, score=score
                ))
        combo_out[outcome] = sorted(best_combos, key=lambda g: g.score, reverse=True)[:6]
    return combo_out

def evaluate_gate_on_row(gate: Dict, row: pd.Series) -> bool:
    for cond in gate["conditions"]:
        val = safe_float(row.get(cond["feature"], np.nan))
        thr = cond["thr"]
        if pd.isna(val):
            return False
        if cond["op"] == ">=" and not (val >= thr):
            return False
        if cond["op"] == "<=" and not (val <= thr):
            return False
    return True

def save_model(model: Dict):
    MODEL_JSON.write_text(json.dumps(model, indent=2))

def load_model() -> Optional[Dict]:
    if not MODEL_JSON.exists():
        return None
    return json.loads(MODEL_JSON.read_text())

# =============================
# Snapshot feature generation
# =============================
def apply_snapshot_overrides(daily_hist: pd.DataFrame, snapshot_df: pd.DataFrame) -> pd.DataFrame:
    hist = daily_hist.copy()
    if snapshot_df.empty:
        return hist
    latest_date = hist["date"].max()
    snap_map = dict(zip(snapshot_df["symbol"], snapshot_df["close"]))
    for sym, close in snap_map.items():
        mask = (hist["symbol"] == sym) & (hist["date"] == latest_date)
        if mask.any():
            hist.loc[mask, "close"] = float(close)
            hist.loc[mask, "high"] = np.maximum(hist.loc[mask, "high"], float(close))
            hist.loc[mask, "low"] = np.minimum(hist.loc[mask, "low"], float(close))
        else:
            hist = pd.concat([hist, pd.DataFrame([{
                "date": latest_date, "symbol": sym, "open": float(close), "high": float(close),
                "low": float(close), "close": float(close), "volume": 0.0, "timeframe": "daily"
            }])], ignore_index=True)
    return hist.sort_values(["symbol", "date"])

def canary_from_row(row: pd.Series) -> Tuple[str, float]:
    checks = []
    for feat, direction in [("RSP_SPY__D3", ">"), ("SMH_SPY__D3", ">"), ("IWM_SPY__D3", ">"),
                            ("XLF_SPY__D3", ">"), ("HYG_IEF__D3", ">"), ("SPXS_SVOL__D3", "<"),
                            ("VXX__D3", "<"), ("$VIX__D3", "<")]:
        val = safe_float(row.get(feat, np.nan))
        if pd.isna(val):
            continue
        checks.append((val > 0) if direction == ">" else (val < 0))
    if not checks:
        return "Neutral", 50.0
    pct = 100.0 * float(np.mean(checks))
    if pct >= 65:
        return "Risk-On", pct
    if pct <= 35:
        return "Risk-Off", pct
    return "Neutral", pct

def analyze_current_snapshot(model: Dict, daily_hist: pd.DataFrame, weekly_hist: pd.DataFrame, snapshot_df: pd.DataFrame, use_proxy: bool = True) -> Dict:
    hist_rt = apply_snapshot_overrides(daily_hist, snapshot_df)
    close_df = pivot_close(hist_rt)
    feat_df = build_level_and_trend_features(close_df)
    latest = feat_df.iloc[-1].copy()
    prior = feat_df.iloc[-2].copy() if len(feat_df) > 1 else pd.Series(dtype=float)

    # proxy governance
    if use_proxy and "NYMO_PROXY" in latest.index:
        latest["$NYMO_EFF"] = latest["NYMO_PROXY"]
        latest["$NYSI_EFF"] = latest.get("NYSI_PROXY", np.nan)
    else:
        latest["$NYMO_EFF"] = latest.get("$NYMO", np.nan)
        latest["$NYSI_EFF"] = latest.get("$NYSI", np.nan)

    gate_results = {}
    reasons = []
    for outcome in ["bounce", "repair", "regime", "fall"]:
        singles = model["gates"][outcome]["single"]
        combos = model["gates"][outcome]["combo"]
        passed = []
        for g in singles + combos:
            if evaluate_gate_on_row(g, latest):
                passed.append(g)
        top = sorted(passed, key=lambda x: x["score"], reverse=True)[:3]
        if top:
            prob = float(np.mean([g["hit_rate"] for g in top]))
            base = float(np.mean([g["base_rate"] for g in top]))
            support = int(np.mean([g["support"] for g in top]))
        else:
            prob = model["base_rates"][outcome]
            base = model["base_rates"][outcome]
            support = 0
        gate_results[outcome] = {
            "prob": prob, "base": base, "support": support, "passed": top
        }

    # improvement breadth
    improve_features = ["$NYSI__D3", "$NYMO__D3", "$BPSPX__BBP_D3", "$SPXA50R__D3", "$NYHL__D3", "$TRIN__D3", "$CPCE__D3"]
    improve_hits = 0
    improve_total = 0
    improve_notes = []
    for feat in improve_features:
        val = safe_float(latest.get(feat, np.nan))
        if pd.isna(val):
            continue
        improve_total += 1
        bullish = (val > 0) if feat not in {"$TRIN__D3", "$CPCE__D3"} else (val < 0)
        if bullish:
            improve_hits += 1
        improve_notes.append((feat, val, bullish))
    improve_frac = improve_hits / improve_total if improve_total else np.nan

    canary_label, canary_score = canary_from_row(latest)
    weekly_overlay = build_weekly_overlay(weekly_hist)

    # verdict logic
    b, r, g, f = gate_results["bounce"]["prob"], gate_results["repair"]["prob"], gate_results["regime"]["prob"], gate_results["fall"]["prob"]
    verdict, reason = "HOLD", "Mixed historical gates; no strong edge."
    if ((b >= max(0.50, gate_results["bounce"]["base"] * 1.5)) or (r >= max(0.42, gate_results["repair"]["base"] * 1.5))) and \
       f < max(0.30, gate_results["fall"]["base"] * 1.15) and canary_label != "Risk-Off":
        verdict = "LONG"
        reason = "Bounce / repair probabilities are historically favorable and downside gates are not dominant."
    elif f >= max(0.40, gate_results["fall"]["base"] * 1.5) and b < max(0.35, gate_results["bounce"]["base"] * 1.15) and canary_label == "Risk-Off":
        verdict = "SHORT"
        reason = "Fall gates dominate, long-side bounce / repair odds are weak, and canary is risk-off."
    elif improve_frac >= 0.65 and b >= gate_results["bounce"]["base"] and canary_label != "Risk-Off":
        verdict = "LONG"
        reason = "Improvement breadth is broad even though absolute levels remain imperfect."
    elif canary_label == "Risk-Off" and weekly_overlay.get("available") and safe_float(weekly_overlay.get("pass_fraction", np.nan)) < 0.45:
        verdict = "SHORT"
        reason = "Canary and weekly overlay are both risk-off / weak."

    # build reasons
    for outcome in ["bounce", "repair", "regime", "fall"]:
        for gte in gate_results[outcome]["passed"][:2]:
            reasons.append(f"{outcome.title()} gate: {gte['description']} (hit {gte['hit_rate']:.0%}, base {gte['base_rate']:.0%})")
    if not reasons:
        reasons.append("No strong gate alignment today.")
    if improve_total:
        reasons.append(f"Improvement breadth: {improve_hits}/{improve_total} tracked repair features are improving.")
    reasons.append(f"Canary: {canary_label} ({canary_score:.0f})")
    if weekly_overlay.get("available"):
        reasons.append(f"Weekly regime pass fraction: {weekly_overlay.get('pass_fraction', np.nan):.0%}")

    return {
        "latest_date": str(close_df.index.max().date()),
        "latest_row": latest.to_dict(),
        "prior_row": prior.to_dict() if len(prior) else {},
        "gate_results": gate_results,
        "canary_label": canary_label,
        "canary_score": canary_score,
        "weekly_overlay": weekly_overlay,
        "verdict": verdict,
        "reason": reason,
        "reasons": reasons,
        "improvement_fraction": improve_frac,
        "improvement_notes": improve_notes,
    }

# =============================
# UI helpers
# =============================
def render_signal_box(verdict: str, text: str):
    cls = {"LONG": "signal-long", "SHORT": "signal-short"}.get(verdict, "signal-hold")
    st.markdown(f"<div class='{cls}'><div style='font-size:1.7rem;font-weight:900;'>📌 Daily Verdict: {verdict}</div><div>{text}</div></div>", unsafe_allow_html=True)

def line_for_gate(gate):
    return f"{gate['description']} | hit {gate['hit_rate']:.0%} vs base {gate['base_rate']:.0%} | support {gate['support']}"

def maybe_save_baseline(daily_hist: pd.DataFrame, weekly_hist: pd.DataFrame):
    daily_hist.to_parquet(DAILY_HIST_PARQUET, index=False)
    if not weekly_hist.empty:
        weekly_hist.to_parquet(WEEKLY_HIST_PARQUET, index=False)

# =============================
# Build model
# =============================
def build_model_from_history(daily_hist: pd.DataFrame, weekly_hist: pd.DataFrame) -> Dict:
    close_df = pivot_close(daily_hist)
    if "RSP" not in close_df.columns:
        raise ValueError("Historical ZIP must include RSP daily history.")
    feat_df = build_level_and_trend_features(close_df)
    full_df = add_forward_outcomes(close_df, feat_df).dropna(subset=["bounce", "repair", "regime", "fall"], how="any")
    # Only use features likely to matter
    candidate_feats = []
    for col in full_df.columns:
        if col in OUTCOME_SPECS:
            continue
        if "__ROC" in col or "__D" in col or "__BBP" in col or "__RSI14" in col or "__CCI20" in col:
            candidate_feats.append(col)
        elif col in PRIMARY_FEATURES:
            candidate_feats.append(col)
    candidate_feats = [c for c in candidate_feats if full_df[c].notna().sum() >= MIN_GATE_SUPPORT * 2]
    single = learn_single_feature_gates(full_df, candidate_feats)
    combo = learn_combo_gates(full_df, single)

    model = {
        "created_at": str(pd.Timestamp.now()),
        "base_rates": {k: float(full_df[k].mean()) for k in OUTCOME_SPECS.keys()},
        "gates": {k: {"single": [asdict(g) for g in single[k]], "combo": [asdict(g) for g in combo[k]]} for k in OUTCOME_SPECS.keys()},
        "features_used": candidate_feats,
        "daily_rows": int(len(daily_hist)),
        "weekly_rows": int(len(weekly_hist)),
        "symbols_daily": sorted(daily_hist["symbol"].unique().tolist()),
        "symbols_weekly": sorted(weekly_hist["symbol"].unique().tolist()) if not weekly_hist.empty else [],
    }
    save_model(model)
    maybe_save_baseline(daily_hist, weekly_hist)
    return model

# =============================
# Main app
# =============================
st.title("Breadth Historical Gate Engine Pro")
st.caption("Historically learned hard gates for bounce / repair / regime / fall, plus improvement-analysis and a single LONG / SHORT / HOLD verdict.")

with st.sidebar:
    st.subheader("Model Controls")
    use_proxy = st.toggle("Use proxy NYMO/NYSI before official evening data", value=True)
    force_rebuild = st.button("Force rebuild historical model")
    reset_model = st.button("Reset saved model")
    if reset_model:
        for p in [MODEL_JSON, DAILY_HIST_PARQUET, WEEKLY_HIST_PARQUET]:
            if p.exists():
                p.unlink()
        st.success("Saved model reset.")

hist_upload = st.file_uploader("One-time historical upload (.zip)", type=["zip"])
snap_upload = st.file_uploader("Daily snapshot upload (.csv)", type=["csv"])

model = load_model()
daily_hist = pd.read_parquet(DAILY_HIST_PARQUET) if DAILY_HIST_PARQUET.exists() else pd.DataFrame()
weekly_hist = pd.read_parquet(WEEKLY_HIST_PARQUET) if WEEKLY_HIST_PARQUET.exists() else pd.DataFrame()

if hist_upload is not None:
    try:
        daily_hist, weekly_hist = parse_stockcharts_zip(hist_upload.read())
        if force_rebuild or model is None:
            with st.spinner("Learning historical gates from uploaded history..."):
                model = build_model_from_history(daily_hist, weekly_hist)
        else:
            maybe_save_baseline(daily_hist, weekly_hist)
            model = load_model()
        st.success("Historical ZIP parsed and model saved.")
    except Exception as e:
        st.error(f"Historical upload failed: {e}")

if model is None:
    st.info("Upload your historical StockCharts ZIP once to build the gate model.")
    st.stop()

if hist_upload is None and DAILY_HIST_PARQUET.exists():
    st.success("Using saved historical gate model. Historical upload is not required again unless you want to refresh the model.")

if daily_hist.empty and DAILY_HIST_PARQUET.exists():
    daily_hist = pd.read_parquet(DAILY_HIST_PARQUET)
if weekly_hist.empty and WEEKLY_HIST_PARQUET.exists():
    weekly_hist = pd.read_parquet(WEEKLY_HIST_PARQUET)

snapshot_df = pd.DataFrame()
if snap_upload is not None:
    try:
        snapshot_df = parse_snapshot_csv(snap_upload.read())
        append_upload_history({"upload_ts": str(pd.Timestamp.now()), "rows": int(len(snapshot_df))})
    except Exception as e:
        st.error(f"Snapshot upload failed: {e}")

analysis = analyze_current_snapshot(model, daily_hist, weekly_hist, snapshot_df, use_proxy=use_proxy)

render_signal_box(analysis["verdict"], analysis["reason"])

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Bounce probability", f"{analysis['gate_results']['bounce']['prob']:.0%}", delta=f"base {analysis['gate_results']['bounce']['base']:.0%}")
with c2:
    st.metric("Repair probability", f"{analysis['gate_results']['repair']['prob']:.0%}", delta=f"base {analysis['gate_results']['repair']['base']:.0%}")
with c3:
    st.metric("Regime probability", f"{analysis['gate_results']['regime']['prob']:.0%}", delta=f"base {analysis['gate_results']['regime']['base']:.0%}")
with c4:
    st.metric("Fall probability", f"{analysis['gate_results']['fall']['prob']:.0%}", delta=f"base {analysis['gate_results']['fall']['base']:.0%}")

c5, c6, c7, c8 = st.columns(4)
with c5:
    st.metric("Canary", analysis["canary_label"], f"{analysis['canary_score']:.0f}")
with c6:
    st.metric("Proxy NYMO", f"{safe_float(analysis['latest_row'].get('$NYMO_EFF', np.nan)):.2f}")
with c7:
    st.metric("Proxy NYSI", f"{safe_float(analysis['latest_row'].get('$NYSI_EFF', np.nan)):.2f}")
with c8:
    pf = analysis["weekly_overlay"].get("pass_fraction", np.nan)
    st.metric("Weekly pass fraction", "n/a" if pd.isna(pf) else f"{pf:.0%}")

tab1, tab2, tab3 = st.tabs(["Decision Dashboard", "Learned Gates", "History / Uploads"])

with tab1:
    left, right = st.columns([1.2, 1.0])
    with left:
        st.markdown("### Why")
        for r in analysis["reasons"]:
            st.write(f"• {r}")
        st.markdown("### Gate Inputs")
        gate_input_lines = []
        for feat in ["$BPSPX__BBP", "$BPSPX", "$SPXA50R", "$NYMO_EFF", "$NYSI_EFF", "$NYHL", "$TRIN", "$CPCE", "$VIX", "VXX", "RSP_SPY"]:
            val = analysis["latest_row"].get(feat, np.nan)
            if pd.notna(val):
                gate_input_lines.append((feat, val))
        if gate_input_lines:
            st.dataframe(pd.DataFrame(gate_input_lines, columns=["Feature", "Value"]), use_container_width=True, hide_index=True)
        else:
            st.info("No strong gate alignment today.")
        st.markdown("### Improvement Analysis")
        imp = pd.DataFrame(analysis["improvement_notes"], columns=["Feature", "Delta", "Bullish?"])
        if not imp.empty:
            st.dataframe(imp, use_container_width=True, hide_index=True)
        else:
            st.info("No improvement metrics were available.")
    with right:
        st.markdown("### Current Snapshot Core Readings")
        core_keys = ["$BPSPX", "$BPSPX__BBP", "$SPXA50R", "$NYMO", "$NYSI", "$NYAD", "$SPXADP", "$NYHL", "$TRIN", "$CPCE", "$VIX", "VXX", "RSP_SPY"]
        core = []
        for k in core_keys:
            v = analysis["latest_row"].get(k, np.nan)
            if pd.notna(v):
                core.append((k, v))
        st.dataframe(pd.DataFrame(core, columns=["Feature", "Current"]), use_container_width=True, hide_index=True)

        chart_symbol = st.selectbox("Breadth chart", ["$SPXA50R", "$BPSPX", "$NYMO", "$NYSI", "$NYHL", "$TRIN", "$CPCE", "$VIX", "VXX", "RSP"])
        close_df = pivot_close(apply_snapshot_overrides(daily_hist, snapshot_df))
        feat_df = build_level_and_trend_features(close_df)
        fig = go.Figure()
        if chart_symbol in close_df.columns:
            fig.add_trace(go.Scatter(x=close_df.index, y=close_df[chart_symbol], name=chart_symbol))
        bcol = f"{chart_symbol}__BBP"
        if bcol in feat_df.columns:
            fig.add_trace(go.Scatter(x=feat_df.index, y=feat_df[bcol], name="%B", yaxis="y2"))
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=25, b=10), yaxis2=dict(overlaying="y", side="right"))
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    for outcome in ["bounce", "repair", "regime", "fall"]:
        st.markdown(f"### {outcome.title()} gates")
        cols = st.columns(2)
        with cols[0]:
            st.markdown("**Best single gates**")
            singles = model["gates"][outcome]["single"]
            if singles:
                st.dataframe(pd.DataFrame([{
                    "Gate": g["description"], "Hit": f"{g['hit_rate']:.0%}", "Base": f"{g['base_rate']:.0%}",
                    "Lift": round(g["lift"], 2), "Support": g["support"]
                } for g in singles]), use_container_width=True, hide_index=True)
            else:
                st.info("No single gates found.")
        with cols[1]:
            st.markdown("**Best combo gates**")
            combos = model["gates"][outcome]["combo"]
            if combos:
                st.dataframe(pd.DataFrame([{
                    "Gate": g["description"], "Hit": f"{g['hit_rate']:.0%}", "Base": f"{g['base_rate']:.0%}",
                    "Lift": round(g["lift"], 2), "Support": g["support"]
                } for g in combos]), use_container_width=True, hide_index=True)
            else:
                st.info("No combo gates found.")

with tab3:
    st.markdown("### Saved model summary")
    st.json({
        "created_at": model["created_at"],
        "daily_rows": model["daily_rows"],
        "weekly_rows": model["weekly_rows"],
        "symbols_daily_count": len(model["symbols_daily"]),
        "symbols_weekly_count": len(model["symbols_weekly"]),
        "latest_baseline_date": analysis["latest_date"],
    })
    if UPLOAD_HISTORY_CSV.exists():
        hist = pd.read_csv(UPLOAD_HISTORY_CSV)
        st.markdown("### Snapshot uploads")
        st.dataframe(hist.tail(20), use_container_width=True, hide_index=True)
