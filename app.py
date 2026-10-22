import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

# ==============================================================================
# 1. PAGE CONFIG & STITCH TACTICAL SCIFI GLASS-BRUTALISM UI
# ==============================================================================
st.set_page_config(
    page_title="Goose Bay Radar HUD // DEFENSE COMMAND",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Stitch Design Tokens & Scifi Glass-Brutalism Styles
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700;800&family=Space+Grotesk:wght@600;700;800&display=swap" rel="stylesheet">

<style>
    /* Base Tactical Canvas */
    .stApp {
        background-color: #090d16 !important;
        background-image: 
            radial-gradient(ellipse at 50% 0%, rgba(56, 189, 248, 0.08) 0%, transparent 60%),
            linear-gradient(rgba(56, 189, 248, 0.035) 1px, transparent 1px),
            linear-gradient(90deg, rgba(56, 189, 248, 0.035) 1px, transparent 1px) !important;
        background-size: 100% 100%, 28px 28px, 28px 28px !important;
        font-family: 'JetBrains Mono', monospace !important;
        color: #dfe2ef !important;
    }
    
    /* Global Typography Override */
    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
    }
    div, p, span, label, input, button {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Fixed Top Telemetry Mast */
    .telemetry-mast {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(10, 14, 23, 0.94);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-top: 2px solid #00ff9d;
        padding: 8px 18px;
        font-size: 0.72rem;
        color: #94a3b8;
        letter-spacing: 0.12em;
        margin-bottom: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.7);
    }
    .mast-tag {
        color: #00ff9d;
        font-weight: 700;
        text-shadow: 0 0 8px rgba(0, 255, 157, 0.5);
    }
    .mast-alert {
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #00ff9d;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse-dot 1.4s infinite;
    }
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; transform: scale(1); box-shadow: 0 0 6px #00ff9d; }
        50% { opacity: 0.3; transform: scale(0.85); box-shadow: none; }
    }

    /* Header Banner */
    .command-header {
        position: relative;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.92) 0%, rgba(10, 14, 23, 0.98) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        padding: 16px 22px;
        margin-bottom: 16px;
        clip-path: polygon(0 0, calc(100% - 14px) 0, 100% 14px, 100% 100%, 0 100%);
        box-shadow: inset 0 0 30px rgba(56, 189, 248, 0.04), 0 8px 30px rgba(0, 0, 0, 0.8);
    }
    .command-header::before {
        content: "";
        position: absolute;
        top: 0; left: 0;
        width: 6px; height: 100%;
        background: linear-gradient(180deg, #00ff9d, #38bdf8);
        box-shadow: 0 0 12px #00ff9d;
    }
    .command-header h1 {
        margin: 0;
        font-size: 1.65rem;
        font-weight: 800;
        color: #f8fafc;
        background: linear-gradient(90deg, #ffffff 0%, #7bd0ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 0.06em;
    }
    .command-header p {
        margin: 4px 0 0 0;
        font-size: 0.82rem;
        color: #94a3b8;
    }

    /* Tactical HUD Metric Cards */
    [data-testid="stMetric"] {
        position: relative;
        background: rgba(13, 19, 31, 0.75) !important;
        border: 1px solid rgba(56, 189, 248, 0.22) !important;
        border-radius: 0px !important;
        padding: 12px 16px !important;
        box-shadow: inset 0 0 15px rgba(56, 189, 248, 0.03), 0 4px 15px rgba(0, 0, 0, 0.5) !important;
        backdrop-filter: blur(10px) !important;
    }
    [data-testid="stMetric"]::before {
        content: "";
        position: absolute;
        top: 0; left: 0;
        width: 10px; height: 2px;
        background: #00ff9d;
        box-shadow: 0 0 6px #00ff9d;
    }
    [data-testid="stMetric"]::after {
        content: "";
        position: absolute;
        bottom: 0; right: 0;
        width: 10px; height: 2px;
        background: #38bdf8;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.72rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.12em !important;
        color: #94a3b8 !important;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1.45rem !important;
        font-weight: 700 !important;
        color: #f8fafc !important;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.2) !important;
    }

    /* Verdict HUD Banner */
    .verdict-box {
        position: relative;
        padding: 18px 20px;
        margin-bottom: 14px;
        text-align: left;
        backdrop-filter: blur(12px);
    }
    .verdict-good {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.7) 0%, rgba(13, 19, 31, 0.95) 100%);
        border: 1px solid #10b981;
        border-left: 5px solid #00ff9d;
        box-shadow: inset 0 0 25px rgba(0, 255, 157, 0.12), 0 0 25px rgba(0, 255, 157, 0.2);
    }
    .verdict-bad {
        background: linear-gradient(135deg, rgba(127, 29, 29, 0.7) 0%, rgba(13, 19, 31, 0.95) 100%);
        border: 1px solid #ef4444;
        border-left: 5px solid #ef4444;
        box-shadow: inset 0 0 25px rgba(239, 68, 68, 0.12), 0 0 25px rgba(239, 68, 68, 0.2);
    }

    /* Tactical Multi-Classifier Box */
    .classifier-telemetry {
        background: rgba(13, 19, 31, 0.85);
        border: 1px solid rgba(56, 189, 248, 0.2);
        padding: 14px 18px;
        margin-top: 10px;
    }
    .classifier-header {
        font-size: 0.75rem;
        letter-spacing: 0.14em;
        color: #7bd0ff;
        text-transform: uppercase;
        margin-bottom: 10px;
        font-weight: 700;
        border-bottom: 1px dashed rgba(56, 189, 248, 0.2);
        padding-bottom: 6px;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: rgba(10, 14, 23, 0.8);
        padding: 6px 8px;
        border: 1px solid rgba(56, 189, 248, 0.2);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 0px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        color: #94a3b8 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        padding: 8px 16px !important;
        border: 1px solid transparent !important;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(56, 189, 248, 0.15) !important;
        color: #00ff9d !important;
        border: 1px solid rgba(0, 255, 157, 0.4) !important;
        text-shadow: 0 0 8px rgba(0, 255, 157, 0.4) !important;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. DATA PIPELINE & PHYSICS FEATURE ENGINEERING
# ==============================================================================
def compute_physics_features(df_pulses: pd.DataFrame) -> pd.DataFrame:
    df_feat = df_pulses.copy()
    real_cols = [f"a{i:02d}" for i in range(1, 35, 2)]
    imag_cols = [f"a{i:02d}" for i in range(2, 35, 2)]

    R = df_feat[real_cols].values.astype(float)
    I = df_feat[imag_cols].values.astype(float)

    # Complex Envelope Magnitude: sqrt(Re^2 + Im^2) across 17 pulse lags
    amplitudes = np.sqrt(R**2 + I**2)
    df_feat["mean_pulse_amplitude"] = np.mean(amplitudes, axis=1)
    df_feat["std_pulse_amplitude"] = np.std(amplitudes, axis=1)
    df_feat["lag_smoothness"] = np.mean(np.abs(np.diff(R, axis=1)), axis=1)

    # Physics Rule: Drop Column 2 (a02) - Lag 0 Imaginary component is 0.0 variance
    if "a02" in df_feat.columns:
        df_feat.drop(columns=["a02"], inplace=True)

    return df_feat


@st.cache_resource(show_spinner="Locking 16-Antenna Beam-Forming Matrix & Calibrating 4 ML Classifiers...")
def build_radar_ml_engine():
    cols = [f"a{i:02d}" for i in range(1, 35)] + ["target"]
    local_path = "ionosphere.csv"
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv"

    if os.path.exists(local_path):
        raw_df = pd.read_csv(local_path, header=None, names=cols)
    else:
        try:
            raw_df = pd.read_csv(url, header=None, names=cols)
        except Exception:
            data = fetch_openml(name="ionosphere", version=1, as_frame=True)
            raw_df = data.frame.copy()
            raw_df.columns = cols

    y = raw_df["target"].astype(str).str.lower().map({"g": 1, "b": 0, "1": 1, "0": 0}).astype(int)
    raw_df["Class_Name"] = np.where(y == 1, "Good (Valid Target)", "Bad (Ghost/Noise)")

    pulse_cols = [f"a{i:02d}" for i in range(1, 35)]
    raw_df[pulse_cols] = raw_df[pulse_cols].astype(float)

    X = compute_physics_features(raw_df[pulse_cols])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Tuned RBF-SVM Pipeline
    svm_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, random_state=42))
    ])
    grid = GridSearchCV(
        svm_pipe,
        {"clf__C": [1, 2, 5, 10, 20], "clf__gamma": ["scale", "auto", 0.01, 0.03, 0.05]},
        cv=cv, scoring="f1", n_jobs=-1
    )
    grid.fit(X_train, y_train)

    models = {
        "SVM (Tuned RBF)": grid.best_estimator_,
        "Neural Network (MLP)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", max_iter=500, random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
        ]),
        "k-NN (k=3)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=3))
        ]),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=42, max_iter=1000))
        ])
    }

    bench_rows, roc_curves, preds_dict = [], {}, {}
    for name, pipe in models.items():
        if name != "SVM (Tuned RBF)":
            pipe.fit(X_train, y_train)
        cv_acc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy").mean()
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        fpr, tpr, _ = roc_curve(y_test, y_prob)

        bench_rows.append({
            "Classifier": name,
            "5-Fold CV (%)": round(cv_acc * 100, 2),
            "Test Accuracy (%)": round(accuracy_score(y_test, y_pred) * 100, 2),
            "Precision (%)": round(precision_score(y_test, y_pred) * 100, 2),
            "Recall (%)": round(recall_score(y_test, y_pred) * 100, 2),
            "F1-Score (%)": round(f1_score(y_test, y_pred) * 100, 2),
            "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
            "Ghost Alarms (FP)": int(fp),
            "Missed Targets (FN)": int(fn)
        })
        roc_curves[name] = (fpr, tpr, roc_auc_score(y_test, y_prob))
        preds_dict[name] = (y_pred, y_prob)

    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    return {
        "raw_df": raw_df, "X": X, "y": y,
        "X_test": X_test, "y_test": y_test,
        "models": models, "best_params": grid.best_params_,
        "bench_df": pd.DataFrame(bench_rows),
        "roc_curves": roc_curves, "preds_dict": preds_dict,
        "X_pca": X_pca, "pca_var": pca.explained_variance_ratio_
    }


state = build_radar_ml_engine()
raw_df, X, y, models = state["raw_df"], state["X"], state["y"], state["models"]


# ==============================================================================
# 3. SIDEBAR CONTROLS
# ==============================================================================
with st.sidebar:
    st.markdown("### 🎛️ RADAR TELEMETRY DECK")
    class_filter = st.radio(
        "FILTER RETURNS:",
        ["All Returns (351)", "Valid Targets (Good)", "Ghost Noise (Bad)"],
        index=0
    )
    if "Valid" in class_filter:
        indices = raw_df[y == 1].index.tolist()
    elif "Ghost" in class_filter:
        indices = raw_df[y == 0].index.tolist()
    else:
        indices = raw_df.index.tolist()

    default_idx = indices.index(148) if 148 in indices else 0
    signal_idx = st.selectbox(
        "RADAR RETURN INDEX:",
        options=indices,
        index=default_idx,
        format_func=lambda i: f"SIGNAL #{i:03d} — {'[TARGET]' if y.iloc[i]==1 else '[GHOST]'}"
    )

    st.divider()
    st.markdown("### ⚡ ELECTRONIC WARFARE SIMULATOR")
    noise_sigma = st.slider("Atmospheric Noise Injection (σ)", 0.0, 0.80, 0.0, 0.05)
    attenuation_pct = st.slider("Ionospheric Attenuation (%)", 0, 80, 0, 5)
    threshold_tau = st.slider("Target Lock Threshold (τ)", 0.10, 0.90, 0.50, 0.05)

    st.divider()
    st.caption("🛡️ **Goose Bay Phased Array:** 16 log-periodic dipole antennas, 8–20 MHz, 6.4 kW peak power pulse radar.")


# ==============================================================================
# 4. SIGNAL PROCESSING FOR SELECTED RETURN
# ==============================================================================
pulse_cols = [f"a{i:02d}" for i in range(1, 35)]
real_cols = [f"a{i:02d}" for i in range(1, 35, 2)]
imag_cols = [f"a{i:02d}" for i in range(2, 35, 2)]

row_raw = raw_df.loc[[signal_idx], pulse_cols].copy()
R_val = row_raw[real_cols].values.flatten().astype(float)
I_val = row_raw[imag_cols].values.flatten().astype(float)

rng = np.random.default_rng(seed=signal_idx + 100)
scale = 1.0 - (attenuation_pct / 100.0)
R_live = np.clip(R_val * scale + rng.normal(0, noise_sigma, 17), -1.0, 1.0)
I_live = np.clip(I_val * scale + rng.normal(0, noise_sigma, 17), -1.0, 1.0)
I_live[0] = 0.0  # Lag 0 Imaginary (a02) is physically locked at 0.0

mod_row = row_raw.copy()
mod_row[real_cols] = R_live
mod_row[imag_cols] = I_live

live_X = compute_physics_features(mod_row)
true_label = int(y.iloc[signal_idx])

svm_prob = float(models["SVM (Tuned RBF)"].predict_proba(live_X)[0][1])
svm_pred = 1 if svm_prob >= threshold_tau else 0
svm_conf = svm_prob if svm_pred == 1 else (1.0 - svm_prob)


# ==============================================================================
# 5. TOP TELEMETRY MAST & HEADER
# ==============================================================================
st.markdown(f"""
<div class="telemetry-mast">
    <div><span class="mast-alert"></span> STATION: <span class="mast-tag">GOOSE BAY LABRADOR (53.3°N 60.4°W)</span></div>
    <div>ARRAY: <span class="mast-tag">16-ANTENNA PHASED 6.4kW</span></div>
    <div>FREQ: <span class="mast-tag">12.45 MHz HF</span></div>
    <div>DEFCON: <span class="mast-tag" style="color:#00ff9d;">DEFCON 3 // ELEVATED</span></div>
    <div>LOCK CUTOFF: <span class="mast-tag">τ = {threshold_tau:.2f}</span></div>
</div>

<div class="command-header">
    <h1>📡 IONOSPHERIC RADAR TARGET VALIDATION HUD</h1>
    <p>SUPERDARN HIGH-LATITUDE ATMOSPHERIC PHENOMENA SURVEILLANCE & TARGET LOCK CONSOLE</p>
</div>
""", unsafe_allow_html=True)

# Telemetry Metrics Grid
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Selected Return", f"SIG #{signal_idx:03d}", "ARRAY 16-DIPOLE")
m2.metric("Ground Truth", "GOOD (TARGET)" if true_label == 1 else "BAD (GHOST/NOISE)", "UCI IONOSPHERE")
m3.metric("Pulse Energy", f"{live_X['mean_pulse_amplitude'].values[0]:.3f}", "√(Re² + Im²)")
m4.metric("Roughness", f"{live_X['lag_smoothness'].values[0]:.3f}", "LAG-TO-LAG JUMP")
m5.metric("Target Probability", f"{svm_prob*100:.1f}%", f"THRESHOLD = {threshold_tau:.2f}")

st.write("")

# ==============================================================================
# 6. NAVIGATION TABS (ORGANIZED DEFENSE WORKSTATION)
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 1. Tactical Radar HUD & Multi-Classifier Verdict",
    "📊 2. Machine Learning Benchmark Suite & ROC Curves",
    "🧬 3. 3D PCA Cluster Separation & Physics Features",
    "🧪 4. Test on New / Unseen Radar Data"
])

# ------------------------------------------------------------------------------
# TAB 1: 360° CANVAS RADAR SWEEP + 17-LAG OSCILLOSCOPE + VERDICT
# ------------------------------------------------------------------------------
with tab1:
    col_scope, col_wave, col_verdict = st.columns([1.15, 1.35, 1.1])

    # 1.1 LIVE HTML5 CANVAS CIRCULAR ROTATING RADAR SWEEP
    with col_scope:
        lags = np.arange(0, 17)
        envelope = np.sqrt(R_live**2 + I_live**2)
        mean_r = float(np.mean(envelope))
        mean_phase = float(np.degrees(np.arctan2(np.mean(I_live), np.mean(R_live))) + 360) % 360
        is_target_int = int(svm_pred)

        radar_canvas_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{ background: #090d16; font-family: 'JetBrains Mono', monospace; overflow: hidden; }}
                .canvas-card {{
                    position: relative;
                    background: rgba(13, 19, 31, 0.9);
                    border: 1px solid rgba(56, 189, 248, 0.25);
                    padding: 8px;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                }}
                .scope-header {{
                    width: 100%;
                    display: flex;
                    justify-content: space-between;
                    color: #7bd0ff;
                    font-size: 10px;
                    font-weight: 700;
                    letter-spacing: 0.12em;
                    margin-bottom: 6px;
                }}
                canvas {{
                    background: radial-gradient(circle at center, #07151e 0%, #060a12 70%, #030509 100%);
                    box-shadow: inset 0 0 20px rgba(0, 255, 157, 0.15), 0 0 15px rgba(0, 0, 0, 0.9);
                    border: 1px solid rgba(56, 189, 248, 0.3);
                }}
            </style>
        </head>
        <body>
            <div class="canvas-card">
                <div class="scope-header">
                    <span>360° POLAR SWEEP RETICLE</span>
                    <span style="color:{'#00ff9d' if is_target_int==1 else '#ef4444'};">{'[LOCKED]' if is_target_int==1 else '[CLUTTER]'}</span>
                </div>
                <canvas id="radarCanvas" width="310" height="310"></canvas>
            </div>
            <script>
                const canvas = document.getElementById('radarCanvas');
                const ctx = canvas.getContext('2d');
                const cx = canvas.width / 2;
                const cy = canvas.height / 2;
                const radius = cx - 12;

                let angle = 0;
                const isTarget = {is_target_int};
                const blipDist = Math.min({mean_r} * (radius * 0.72), radius - 15);
                const blipAngle = ({mean_phase} * Math.PI) / 180;

                function draw() {{
                    ctx.clearRect(0, 0, canvas.width, canvas.height);

                    // Concentric Range Rings
                    const rings = [0.25, 0.5, 0.75, 1.0];
                    ctx.strokeStyle = "rgba(56, 189, 248, 0.2)";
                    ctx.lineWidth = 1;
                    rings.forEach((r, idx) => {{
                        ctx.beginPath();
                        ctx.arc(cx, cy, radius * r, 0, Math.PI * 2);
                        ctx.stroke();
                        // Ring labels
                        ctx.fillStyle = "rgba(148, 163, 184, 0.5)";
                        ctx.font = "8px monospace";
                        ctx.fillText((idx + 1) * 250 + "km", cx + 4, cy - (radius * r) + 10);
                    }});

                    // Crosshair Azimuth lines
                    ctx.strokeStyle = "rgba(56, 189, 248, 0.25)";
                    ctx.beginPath();
                    ctx.moveTo(cx - radius, cy); ctx.lineTo(cx + radius, cy);
                    ctx.moveTo(cx, cy - radius); ctx.lineTo(cx, cy + radius);
                    ctx.stroke();

                    // Dial Headings
                    ctx.fillStyle = "#38bdf8";
                    ctx.font = "9px monospace";
                    ctx.fillText("000°", cx - 10, cy - radius + 11);
                    ctx.fillText("090°", cx + radius - 26, cy + 3);
                    ctx.fillText("180°", cx - 10, cy + radius - 4);
                    ctx.fillText("270°", cx - radius + 4, cy + 3);

                    // Rotating Beam Sweep Trail
                    const sweepGradient = ctx.createRadialGradient(cx, cy, 0, cx, cy, radius);
                    const beamColor = isTarget === 1 ? "rgba(0, 255, 157, " : "rgba(239, 68, 68, ";
                    
                    ctx.save();
                    ctx.beginPath();
                    ctx.moveTo(cx, cy);
                    ctx.arc(cx, cy, radius, angle - 0.45, angle);
                    ctx.closePath();
                    ctx.fillStyle = beamColor + "0.22)";
                    ctx.fill();
                    ctx.restore();

                    // Main Sweep Ray
                    ctx.strokeStyle = isTarget === 1 ? "#00ff9d" : "#ef4444";
                    ctx.lineWidth = 1.8;
                    ctx.shadowColor = isTarget === 1 ? "#00ff9d" : "#ef4444";
                    ctx.shadowBlur = 8;
                    ctx.beginPath();
                    ctx.moveTo(cx, cy);
                    ctx.lineTo(cx + radius * Math.cos(angle), cy + radius * Math.sin(angle));
                    ctx.stroke();
                    ctx.shadowBlur = 0;

                    // Plot Radar Return Blip
                    const bx = cx + blipDist * Math.cos(blipAngle);
                    const by = cy + blipDist * Math.sin(blipAngle);

                    if (isTarget === 1) {{
                        // Solid Pulsing Phosphor Locked Blip
                        ctx.fillStyle = "#00ff9d";
                        ctx.shadowColor = "#00ff9d";
                        ctx.shadowBlur = 12;
                        ctx.beginPath();
                        ctx.arc(bx, by, 5.5, 0, Math.PI * 2);
                        ctx.fill();

                        // Target Acquisition Reticle Box
                        ctx.strokeStyle = "#38bdf8";
                        ctx.lineWidth = 1.2;
                        ctx.strokeRect(bx - 9, by - 9, 18, 18);

                        ctx.fillStyle = "#ffffff";
                        ctx.font = "8px monospace";
                        ctx.fillText("LOCK #"+{signal_idx}, bx + 12, by - 2);
                    }} else {{
                        // Flickering Ghost Noise Blip
                        const flicker = Math.random() > 0.35 ? 1 : 0.2;
                        ctx.fillStyle = "rgba(239, 68, 68, " + flicker + ")";
                        ctx.shadowColor = "#ef4444";
                        ctx.shadowBlur = 8;
                        ctx.beginPath();
                        ctx.arc(bx, by, 4.5, 0, Math.PI * 2);
                        ctx.fill();

                        ctx.fillStyle = "rgba(239, 68, 68, " + flicker + ")";
                        ctx.font = "8px monospace";
                        ctx.fillText("GHOST", bx + 10, by - 2);
                    }}

                    angle = (angle + 0.035) % (Math.PI * 2);
                    requestAnimationFrame(draw);
                }}
                draw();
            </script>
        </body>
        </html>
        """
        components.html(radar_canvas_html, height=360)

    # 1.2 17-LAG COMPLEX OSCILLOSCOPE (DARK GLASS PLOTLY)
    with col_wave:
        fig_osc = go.Figure()
        fig_osc.add_trace(go.Scatter(
            x=lags, y=envelope, mode="lines",
            name="Envelope √(Re²+Im²)",
            line=dict(color="#00ff9d", width=1.8, dash="dot"),
            fill="tozeroy", fillcolor="rgba(0, 255, 157, 0.08)"
        ))
        fig_osc.add_trace(go.Scatter(
            x=lags, y=R_live, mode="lines+markers",
            name="Real (In-Phase)",
            line=dict(color="#38bdf8", width=2.8),
            marker=dict(size=7, color="#38bdf8", line=dict(width=1, color="white"))
        ))
        fig_osc.add_trace(go.Scatter(
            x=lags, y=I_live, mode="lines+markers",
            name="Imag (Quadrature)",
            line=dict(color="#ef4444", width=2.2, dash="dash"),
            marker=dict(size=6, symbol="square", color="#ef4444")
        ))
        fig_osc.update_layout(
            title="<b>17-LAG COMPLEX AUTOCORRELATION WAVEFORM</b>",
            xaxis=dict(title="Radar Pulse Lag (k = 0 to 16)", tickmode="linear", dtick=1, gridcolor="rgba(56, 189, 248, 0.12)"),
            yaxis=dict(title="Signal Return [-1, +1]", range=[-1.2, 1.3], gridcolor="rgba(56, 189, 248, 0.12)"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10, 14, 23, 0.8)",
            font=dict(color="#dfe2ef", family="JetBrains Mono"),
            height=360,
            legend=dict(orientation="h", y=1.12, x=0, font=dict(size=10)),
            margin=dict(l=35, r=15, t=50, b=35)
        )
        st.plotly_chart(fig_osc, use_container_width=True)

    # 1.3 VERDICT CARD & ALL 4 ML MODEL GAUGES
    with col_verdict:
        if svm_pred == 1:
            st.markdown(f"""
            <div class="verdict-box verdict-good">
                <div style="font-size:0.75rem; color:#00ff9d; letter-spacing:0.16em; font-weight:700;">AI TARGET CLASSIFIER</div>
                <h3 style="color:#ffffff; margin:4px 0 6px 0; font-size:1.35rem;">✅ VALID TARGET LOCKED</h3>
                <div style="font-size:1.0rem; color:#00ff9d; font-weight:700;">Confidence: {svm_conf*100:.2f}%</div>
                <p style="font-size:0.76rem; color:#cbd5e1; margin-top:6px; line-height:1.4;">
                    Coherent phase decay detected across the 17 pulses. Free electron density reflection confirmed.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-box verdict-bad">
                <div style="font-size:0.75rem; color:#fca5a5; letter-spacing:0.16em; font-weight:700;">AI TARGET CLASSIFIER</div>
                <h3 style="color:#ffffff; margin:4px 0 6px 0; font-size:1.35rem;">🚨 GHOST / NOISE REJECTED</h3>
                <div style="font-size:1.0rem; color:#ef4444; font-weight:700;">Confidence: {svm_conf*100:.2f}%</div>
                <p style="font-size:0.76rem; color:#cbd5e1; margin-top:6px; line-height:1.4;">
                    Erratic phase jumps and zero return power. Atmosphere pass-through clutter filtered to prevent false alarm.
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="classifier-telemetry">
            <div class="classifier-header">4-CLASSIFIER INFERENCE TELEMETRY</div>
        """, unsafe_allow_html=True)

        for m_name, m_pipe in models.items():
            p_val = float(m_pipe.predict_proba(live_X)[0][1])
            is_good = p_val >= threshold_tau
            badge = "🟢 TARGET" if is_good else "🔴 NOISE"
            st.progress(int(p_val * 100), text=f"**{m_name}**: {p_val*100:.1f}% ({badge})")

        st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# TAB 2: BENCHMARK SUITE, CONFUSION MATRIX & ROC CURVES
# ------------------------------------------------------------------------------
with tab2:
    st.markdown("#### 🏆 BENCHMARK EVALUATION (80/20 STRATIFIED SPLIT + 5-FOLD CROSS-VALIDATION)")
    st.dataframe(
        state["bench_df"].style.highlight_max(
            subset=["5-Fold CV (%)", "Test Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)", "ROC-AUC"],
            color="#065f46"
        ).highlight_min(
            subset=["Ghost Alarms (FP)", "Missed Targets (FN)"],
            color="#065f46"
        ),
        use_container_width=True,
        hide_index=True
    )

    c_cm, c_roc = st.columns(2)

    with c_cm:
        _, y_prob_svm = state["preds_dict"]["SVM (Tuned RBF)"]
        y_pred_tau = (y_prob_svm >= threshold_tau).astype(int)
        cm = confusion_matrix(state["y_test"], y_pred_tau)
        tn, fp, fn, tp = cm.ravel()

        z_text = [
            [f"<b>TN = {tn}</b><br>Filtered Noise", f"<b>FP = {fp}</b><br>🚨 Ghost Alarms"],
            [f"<b>FN = {fn}</b><br>⚠️ Missed Targets", f"<b>TP = {tp}</b><br>✅ Locked Targets"]
        ]
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=["PRED: BAD (NOISE)", "PRED: GOOD (TARGET)"],
            y=["ACTUAL: BAD (NOISE)", "ACTUAL: GOOD (TARGET)"],
            colorscale=[[0, "#090d16"], [0.5, "#065f46"], [1, "#00ff9d"]],
            showscale=False
        ))
        for i in range(2):
            for j in range(2):
                fig_cm.add_annotation(
                    x=["PRED: BAD (NOISE)", "PRED: GOOD (TARGET)"][j],
                    y=["ACTUAL: BAD (NOISE)", "ACTUAL: GOOD (TARGET)"][i],
                    text=z_text[i][j], showarrow=False,
                    font=dict(size=14, color="white", family="JetBrains Mono")
                )
        fig_cm.update_layout(
            title=f"<b>CONFUSION MATRIX (TUNED RBF-SVM @ τ = {threshold_tau:.2f})</b>",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10, 14, 23, 0.8)",
            font=dict(color="#dfe2ef", family="JetBrains Mono"),
            height=370,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with c_roc:
        fig_roc = go.Figure()
        palette = {
            "SVM (Tuned RBF)": "#00ff9d",
            "Neural Network (MLP)": "#a855f7",
            "Random Forest": "#38bdf8",
            "k-NN (k=3)": "#f59e0b",
            "Logistic Regression": "#ef4444"
        }
        for name, (fpr, tpr, auc_v) in state["roc_curves"].items():
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"{name} (AUC={auc_v:.3f})",
                line=dict(width=3 if "SVM" in name else 2, color=palette[name])
            ))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Guess", line=dict(dash="dash", color="gray")))
        fig_roc.update_layout(
            title="<b>ROC-AUC CURVES (71 RADAR TEST RETURNS)</b>",
            xaxis=dict(title="False Positive Rate (Ghost Alarms)", gridcolor="rgba(56, 189, 248, 0.12)"),
            yaxis=dict(title="True Positive Rate (Target Recall)", gridcolor="rgba(56, 189, 248, 0.12)"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10, 14, 23, 0.8)",
            font=dict(color="#dfe2ef", family="JetBrains Mono"),
            height=370,
            legend=dict(x=0.45, y=0.15, font=dict(size=10)),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_roc, use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 3: 3D PCA CLUSTER PROOF & PHYSICS FEATURE IMPORTANCE
# ------------------------------------------------------------------------------
with tab3:
    col_pca, col_imp = st.columns([1.2, 1.0])

    with col_pca:
        pca_df = pd.DataFrame({
            "PC1": state["X_pca"][:, 0],
            "PC2": state["X_pca"][:, 1],
            "PC3": state["X_pca"][:, 2],
            "Class": raw_df["Class_Name"],
            "Signal": [f"Signal #{i:03d}" for i in range(len(raw_df))]
        })
        fig_pca = px.scatter_3d(
            pca_df, x="PC1", y="PC2", z="PC3", color="Class", hover_name="Signal",
            color_discrete_map={"Good (Valid Target)": "#00ff9d", "Bad (Ghost/Noise)": "#ef4444"},
            title="<b>3D PCA CLUSTERING: COHERENT TARGETS VS SCATTERED CLUTTER</b>"
        )
        fig_pca.update_traces(marker=dict(size=4.5, opacity=0.85, line=dict(width=0.5, color="white")))
        fig_pca.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#dfe2ef", family="JetBrains Mono"),
            height=430,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig_pca, use_container_width=True)

    with col_imp:
        rf_model = models["Random Forest"].named_steps["clf"]
        imp_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": rf_model.feature_importances_
        }).sort_values("Importance", ascending=False).head(12)

        fig_imp = px.bar(
            imp_df[::-1], x="Importance", y="Feature", orientation="h",
            color="Importance",
            color_continuous_scale=[[0, "#00354a"], [0.5, "#00a6e0"], [1, "#00ff9d"]],
            title="<b>TOP 12 RADAR FEATURES (PHYSICS FEATURES LEAD RANKING)</b>"
        )
        fig_imp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10, 14, 23, 0.8)",
            font=dict(color="#dfe2ef", family="JetBrains Mono"),
            height=430,
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_imp, use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 4: TEST ON BRAND-NEW / UNSEEN RADAR DATA (3 MODES FOR VIVA)
# ------------------------------------------------------------------------------
with tab4:
    st.markdown("### 🧪 TEST CLASSIFIER ON BRAND-NEW, UNSEEN RADAR SIGNALS")
    test_mode = st.radio(
        "CHOOSE HOW TO INPUT NEW DATA:",
        [
            "⚡ Mode 1: Synthetic Radar Pulse Generator (Create New Signal Live)",
            "📋 Mode 2: Paste 34 Raw Comma-Separated Values (Single Row Test)",
            "📂 Mode 3: Upload New CSV File (Batch Unseen Data Test)"
        ],
        horizontal=True
    )
    st.divider()

    # ==========================================================================
    # MODE 1: SYNTHETIC RADAR WAVE GENERATOR (NO DATASET ROWS USED)
    # ==========================================================================
    if "Mode 1" in test_mode:
        g_col1, g_col2 = st.columns([1.0, 1.6])
        with g_col1:
            st.markdown("#### 🎛️ Synthesize a Brand-New 17-Lag Return")
            st.caption("Generates a brand-new 34-feature complex wave using radar physics equations:")
            preset = st.selectbox(
                "Quick Signal Preset:",
                ["Custom Sliders", "Simulated Aircraft/Ionosphere Target (Coherent)", "Simulated Atmospheric Clutter (Ghost Noise)"]
            )
            if preset == "Simulated Aircraft/Ionosphere Target (Coherent)":
                d_rate, doppler_f, clutter_n = 0.08, 0.35, 0.04
            elif preset == "Simulated Atmospheric Clutter (Ghost Noise)":
                d_rate, doppler_f, clutter_n = 0.02, 1.45, 0.65
            else:
                d_rate, doppler_f, clutter_n = 0.10, 0.40, 0.08

            decay = st.slider("Pulse Envelope Decay Rate (α)", 0.0, 0.30, d_rate, 0.01)
            freq = st.slider("Doppler Phase Shift Frequency (ω)", 0.05, 2.00, doppler_f, 0.05)
            clutter = st.slider("Random Clutter / Noise Amplitude", 0.0, 0.90, clutter_n, 0.02)

        # Generate 17 lags (k = 0..16) mathematically: A(k) = exp(-decay*k) * exp(j * freq * k) + noise
        lags_k = np.arange(0, 17)
        rng_syn = np.random.default_rng(seed=int((decay * 100) + (freq * 100) + (clutter * 100)))
        R_syn = np.exp(-decay * lags_k) * np.cos(freq * lags_k) + rng_syn.normal(0, clutter, 17)
        I_syn = np.exp(-decay * lags_k) * np.sin(freq * lags_k) + rng_syn.normal(0, clutter, 17)
        R_syn[0] = 1.0  # Normalized Lag-0 Real power
        I_syn[0] = 0.0  # Lag-0 Imaginary (a02) is always 0.0
        R_syn, I_syn = np.clip(R_syn, -1.0, 1.0), np.clip(I_syn, -1.0, 1.0)

        # Interleave into a01..a34 DataFrame
        syn_dict = {}
        for k in range(17):
            syn_dict[f"a{(2*k)+1:02d}"] = [R_syn[k]]
            syn_dict[f"a{(2*k)+2:02d}"] = [I_syn[k]]
        syn_raw_df = pd.DataFrame(syn_dict)
        syn_X = compute_physics_features(syn_raw_df)

        syn_prob = float(models["SVM (Tuned RBF)"].predict_proba(syn_X)[0][1])
        syn_pred = 1 if syn_prob >= threshold_tau else 0

        with g_col2:
            fig_syn = go.Figure()
            fig_syn.add_trace(go.Scatter(x=lags_k, y=R_syn, mode="lines+markers", name="Real (a01, a03..a33)", line=dict(color="#38bdf8", width=3)))
            fig_syn.add_trace(go.Scatter(x=lags_k, y=I_syn, mode="lines+markers", name="Imaginary (a02, a04..a34)", line=dict(color="#ef4444", width=2.5, dash="dash")))
            fig_syn.update_layout(
                title=f"<b>Newly Synthesized Unseen Signal</b> — Verdict: {'✅ GOOD TARGET' if syn_pred==1 else '🚨 BAD NOISE'} ({syn_prob*100:.1f}% Target Prob)",
                xaxis_title="Pulse Lag (k = 0 to 16)",
                yaxis=dict(title="Amplitude [-1, +1]", range=[-1.2, 1.2], gridcolor="rgba(56, 189, 248, 0.12)"),
                xaxis=dict(gridcolor="rgba(56, 189, 248, 0.12)"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(10, 14, 23, 0.8)",
                font=dict(color="#dfe2ef", family="JetBrains Mono"),
                height=340,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_syn, use_container_width=True)

            st.caption("**Generated 34 Raw Features (`a01`–`a34`) + 3 Engineered Physics Features:**")
            st.dataframe(syn_X, use_container_width=True, hide_index=True)

    # ==========================================================================
    # MODE 2: PASTE 34 COMMA-SEPARATED NUMBERS
    # ==========================================================================
    elif "Mode 2" in test_mode:
        st.markdown("#### 📋 Paste Any 34 Comma-Separated Values (`a01` to `a34`)")
        default_str = "1, 0, 0.92, -0.08, 0.85, -0.15, 0.76, -0.22, 0.65, -0.28, 0.54, -0.31, 0.42, -0.33, 0.31, -0.32, 0.21, -0.29, 0.12, -0.25, 0.04, -0.20, -0.02, -0.15, -0.07, -0.10, -0.10, -0.06, -0.12, -0.03, -0.13, -0.01, -0.12, 0.01"
        raw_text = st.text_area("Enter 34 values between -1.0 and +1.0:", value=default_str, height=90)

        if st.button("🚀 Validate Custom Signal", type="primary"):
            try:
                vals = [float(x.strip()) for x in raw_text.split(",") if x.strip() != ""]
                if len(vals) != 34:
                    st.error(f"Expected exactly 34 numbers, but you provided {len(vals)}.")
                else:
                    custom_df = pd.DataFrame([vals], columns=[f"a{i:02d}" for i in range(1, 35)])
                    custom_X = compute_physics_features(custom_df)
                    c_prob = float(models["SVM (Tuned RBF)"].predict_proba(custom_X)[0][1])
                    c_pred = 1 if c_prob >= threshold_tau else 0

                    if c_pred == 1:
                        st.success(f"✅ **VALID TARGET DETECTED (GOOD)** — Target Probability: **{c_prob*100:.2f}%** | Mean Energy: `{custom_X['mean_pulse_amplitude'].values[0]:.3f}`")
                    else:
                        st.error(f"🚨 **GHOST / NOISE REJECTED (BAD)** — Target Probability: **{c_prob*100:.2f}%** | Lag Roughness: `{custom_X['lag_smoothness'].values[0]:.3f}`")
            except ValueError:
                st.error("Please ensure all 34 values are valid numbers separated by commas.")

    # ==========================================================================
    # MODE 3: UPLOAD UNSEEN CSV FILE (BATCH PREDICTION)
    # ==========================================================================
    else:
        st.markdown("#### 📂 Batch CSV Uploader for Unseen Radar Returns")
        # Generate a downloadable 15-row synthetic unseen CSV for teacher demo
        unseen_sample = state["X_test"].copy()
        unlabeled_test_csv = raw_df.loc[state["X_test"].index[:15], pulse_cols].to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 1. Download Sample Unlabeled Test CSV (15 Unseen Signals)",
            data=unlabeled_test_csv,
            file_name="unseen_radar_test_signals.csv",
            mime="text/csv"
        )

        uploaded_csv = st.file_uploader("📤 2. Upload Any CSV File with 34 Columns (a01..a34)", type=["csv"])
        if uploaded_csv is not None:
            up_df = pd.read_csv(uploaded_csv)
            if up_df.shape[1] >= 34:
                raw_34 = up_df.iloc[:, :34].copy()
                raw_34.columns = [f"a{i:02d}" for i in range(1, 35)]
                feat_36 = compute_physics_features(raw_34)

                probs = models["SVM (Tuned RBF)"].predict_proba(feat_36)[:, 1]
                preds = (probs >= threshold_tau).astype(int)

                out_df = raw_34.copy()
                out_df.insert(0, "Mean_Pulse_Energy", feat_36["mean_pulse_amplitude"].round(3))
                out_df.insert(0, "Target_Probability (%)", (probs * 100).round(2))
                out_df.insert(0, "AI_Verdict", np.where(preds == 1, "GOOD (Valid Target)", "BAD (Ghost/Noise)"))

                st.success(f"✅ Classified {len(out_df)} unseen radar returns: **{(preds==1).sum()} Valid Targets** and **{(preds==0).sum()} Ghost/Noise Signals**.")
                st.dataframe(out_df, use_container_width=True)
            else:
                st.error("Uploaded CSV must have at least 34 columns representing the 17 complex radar pulses.")

