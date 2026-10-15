"""
EXAMIQ — Exam Score Intelligence
Streamlit application featuring OLED dark glassmorphism, live inference using
the saved XGBoost pipeline model, and verified evaluation metrics from Boosting.ipynb.
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Suppress sklearn and xgboost version mismatch warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1. PAGE CONFIGURATION & METADATA
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EXAMIQ — Exam Score Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. DELIVR-INSPIRED OLED GLASSMORPHISM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
    --bg-oled: #050505;
    --bg-card: rgba(14, 15, 24, 0.75);
    --bg-card-sub: rgba(22, 24, 38, 0.6);
    --border-glass: rgba(255, 255, 255, 0.08);
    --border-accent: rgba(99, 102, 241, 0.35);
    --accent-indigo: #6366f1;
    --accent-violet: #8b5cf6;
    --accent-purple: #a855f7;
    --accent-glow: rgba(99, 102, 241, 0.25);
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --success: #10b981;
    --warning: #f59e0b;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-oled) !important;
    background-image: 
        radial-gradient(circle at 18% -8%, rgba(99, 102, 241, 0.18) 0%, transparent 48%),
        radial-gradient(circle at 82% 108%, rgba(139, 92, 246, 0.14) 0%, transparent 52%),
        radial-gradient(circle at 50% 38%, rgba(168, 85, 247, 0.06) 0%, transparent 65%) !important;
    color: var(--text-primary) !important;
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Hide default Streamlit clutter */
#MainMenu {visibility: hidden !important;}
footer {visibility: hidden !important;}
header {visibility: hidden !important;}
[data-testid="stToolbar"] {visibility: hidden !important;}
[data-testid="stDecoration"] {visibility: hidden !important;}
.stDeployButton {display: none !important;}

/* Global padding */
.block-container {
    padding-top: 1.6rem !important;
    padding-bottom: 3.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1260px !important;
}

@media (max-width: 768px) {
    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
}

/* Glass Card Container */
.glass-panel {
    background: var(--bg-card);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--border-glass);
    border-radius: 20px;
    padding: 26px 30px;
    margin-bottom: 24px;
    box-shadow: 0 12px 35px -12px rgba(0, 0, 0, 0.7), 0 0 1px 1px rgba(255, 255, 255, 0.03);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.glass-panel:hover {
    border-color: rgba(99, 102, 241, 0.3);
    box-shadow: 0 16px 45px -12px rgba(99, 102, 241, 0.18);
}

/* Header Section */
.header-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 5px 14px;
    background: rgba(99, 102, 241, 0.12);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #a5b4fc;
    margin-bottom: 12px;
}
.header-badge .dot {
    width: 6px;
    height: 6px;
    background: #34d399;
    border-radius: 50%;
    box-shadow: 0 0 8px #34d399;
    animation: pulseGlow 2s infinite ease-in-out;
}
@keyframes pulseGlow {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.85); }
}

.brand-title {
    font-size: 3.1rem;
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -0.04em;
    background: linear-gradient(135deg, #ffffff 40%, #c7d2fe 75%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 6px 0;
}

.brand-subtitle {
    font-size: 1.2rem;
    font-weight: 600;
    color: #cbd5e1;
    letter-spacing: -0.01em;
    margin-bottom: 4px;
}

.brand-desc {
    font-size: 0.94rem;
    color: var(--text-secondary);
    margin: 0 0 22px 0;
}

/* Metric Cards */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 26px;
}
@media (max-width: 900px) {
    .metrics-grid { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 520px) {
    .metrics-grid { grid-template-columns: 1fr; }
}

.metric-box {
    background: rgba(14, 16, 26, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 16px;
    padding: 18px 22px;
    position: relative;
    overflow: hidden;
    transition: transform 0.25s ease, border-color 0.25s ease;
}
.metric-box:hover {
    transform: translateY(-2px);
    border-color: rgba(99, 102, 241, 0.38);
}
.metric-box::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.65), transparent);
}
.metric-title {
    font-size: 0.73rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--text-muted);
    margin-bottom: 6px;
}
.metric-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.75rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1.2;
}
.metric-sub {
    font-size: 0.72rem;
    color: #94a3b8;
    margin-top: 5px;
}

/* Section Header */
.section-title {
    font-size: 1.26rem;
    font-weight: 700;
    color: #f8fafc;
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 6px;
}
.section-title span.icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    background: rgba(99, 102, 241, 0.16);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 8px;
    font-size: 0.85rem;
    color: #a5b4fc;
}
.section-desc {
    font-size: 0.85rem;
    color: var(--text-secondary);
    margin-bottom: 20px;
}

/* Presets Bar */
.presets-bar {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 14px;
    padding: 12px 18px 8px 18px;
    margin-bottom: 20px;
}
.presets-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 10px;
}
.presets-title {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #818cf8;
}
.presets-hint {
    font-size: 0.74rem;
    color: #64748b;
}

/* BUTTONS — FIXED PROPERLY FOR ALL SCREEN SIZES */
button {
    white-space: nowrap !important;
    word-break: keep-all !important;
    box-sizing: border-box !important;
}

button p, button span, button div {
    white-space: nowrap !important;
    word-break: keep-all !important;
    overflow: visible !important;
    text-align: center !important;
}

/* Secondary Button (Preset / Reset) — Elegant Glass Pill */
button[data-testid="stBaseButton-secondary"],
button[kind="secondary"] {
    background: rgba(255, 255, 255, 0.035) !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    padding: 10px 16px !important;
    font-size: 0.87rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    white-space: nowrap !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) !important;
    backdrop-filter: blur(12px) !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    min-height: 44px !important;
    width: 100% !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

button[data-testid="stBaseButton-secondary"]:hover,
button[kind="secondary"]:hover {
    background: rgba(99, 102, 241, 0.15) !important;
    border-color: rgba(99, 102, 241, 0.45) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 18px rgba(99, 102, 241, 0.25) !important;
    transform: translateY(-1px) !important;
}

button[data-testid="stBaseButton-secondary"]:active,
button[kind="secondary"]:active {
    transform: translateY(1px) !important;
}

/* Primary Button (Calculate Prediction) — High-End Accent Glow */
button[data-testid="stBaseButton-primary"],
button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 60%, #4338ca 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 14px !important;
    padding: 14px 28px !important;
    font-weight: 700 !important;
    font-size: 1.02rem !important;
    letter-spacing: 0.025em !important;
    white-space: nowrap !important;
    box-shadow: 0 6px 24px rgba(79, 70, 229, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.25) !important;
    min-height: 50px !important;
    width: 100% !important;
    transition: all 0.25s ease !important;
}

button[data-testid="stBaseButton-primary"]:hover,
button[kind="primary"]:hover {
    background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%) !important;
    box-shadow: 0 8px 32px rgba(79, 70, 229, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.35) !important;
    transform: translateY(-1px) !important;
}

button[data-testid="stBaseButton-primary"]:active,
button[kind="primary"]:active {
    transform: translateY(1px) !important;
}

/* All Form Widget Labels / Feature Titles — High Contrast & Bright */
div[data-testid="stSelectbox"] label,
div[data-testid="stSelectbox"] label p,
div[data-testid="stSelectbox"] label span,
div[data-testid="stSlider"] label,
div[data-testid="stSlider"] label p,
div[data-testid="stSlider"] label span,
div[data-testid="stWidgetLabel"],
div[data-testid="stWidgetLabel"] p,
div[data-testid="stWidgetLabel"] span,
label p,
label span {
    color: #f1f5f9 !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.015em !important;
    opacity: 1 !important;
    margin-bottom: 6px !important;
    display: inline-block !important;
}

/* Streamlit Select Box Overrides — Fix text clipping at bottom */
div[data-baseweb="select"] {
    border-radius: 12px !important;
}

div[data-baseweb="select"] > div {
    background-color: rgba(18, 20, 32, 0.85) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 12px !important;
    min-height: 48px !important;
    height: 48px !important;
    padding: 0 14px !important;
    display: flex !important;
    align-items: center !important;
    box-sizing: border-box !important;
    transition: border-color 0.2s ease, background-color 0.2s ease !important;
}

div[data-baseweb="select"] > div:hover {
    border-color: rgba(99, 102, 241, 0.55) !important;
    background-color: rgba(24, 27, 44, 0.9) !important;
}

/* Inner selected text container — ensure ample line-height and no bottom clipping */
div[data-baseweb="select"] div[aria-selected="true"],
div[data-baseweb="select"] [role="combobox"],
div[data-baseweb="select"] > div > div,
div[data-baseweb="select"] span {
    color: #ffffff !important;
    font-size: 0.92rem !important;
    font-weight: 500 !important;
    line-height: 1.5 !important;
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}

/* Prevent overflow clipping on selectbox text */
div[data-baseweb="select"] * {
    overflow: visible !important;
}

div[data-baseweb="popover"] ul {
    background-color: #0e101a !important;
    border: 1px solid rgba(99, 102, 241, 0.35) !important;
    border-radius: 12px !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.7) !important;
}

div[data-baseweb="popover"] li {
    color: #f1f5f9 !important;
    font-size: 0.88rem !important;
    padding: 10px 14px !important;
}
div[data-baseweb="popover"] li:hover {
    background-color: rgba(99, 102, 241, 0.15) !important;
}

/* Sliders — Indigo Track, Glowing Thumb, Bright Values */
div[data-testid="stSlider"] [data-baseweb="slider"] {
    margin-top: 6px;
}
div[data-testid="stSlider"] [data-baseweb="slider"] > div > div {
    background-color: #6366f1 !important;
}
div[data-testid="stSlider"] div[role="slider"] {
    background-color: #818cf8 !important;
    border: 2px solid #ffffff !important;
    box-shadow: 0 0 12px rgba(99, 102, 241, 0.75) !important;
    width: 18px !important;
    height: 18px !important;
}
div[data-testid="stSlider"] div[data-testid="stThumbValue"] {
    color: #c7d2fe !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
}


/* Result Card */
.result-card {
    background: linear-gradient(145deg, rgba(20, 22, 38, 0.85), rgba(12, 13, 22, 0.95));
    border: 1px solid rgba(99, 102, 241, 0.35);
    border-radius: 20px;
    padding: 28px;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 12px 35px -10px rgba(99, 102, 241, 0.28);
    margin-top: 14px;
}
.result-card::after {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at center, rgba(99, 102, 241, 0.08) 0%, transparent 60%);
    pointer-events: none;
}
.result-score {
    font-family: 'JetBrains Mono', monospace;
    font-size: 3.8rem;
    font-weight: 800;
    line-height: 1;
    background: linear-gradient(135deg, #ffffff 40%, #c7d2fe 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 12px 0 6px 0;
}
.grade-badge {
    display: inline-block;
    padding: 5px 16px;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.factors-summary {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-top: 18px;
    padding-top: 18px;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
}
@media (max-width: 650px) {
    .factors-summary { grid-template-columns: 1fr; }
}
.factor-chip {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 10px;
    padding: 8px 10px;
}
.factor-chip-title {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 2px;
}
.factor-chip-val {
    font-size: 0.84rem;
    font-weight: 600;
    color: #cbd5e1;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: transparent;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}
.stTabs [data-baseweb="tab"] {
    color: var(--text-muted) !important;
    font-weight: 600;
    font-size: 0.88rem;
    padding: 10px 18px;
    border-radius: 8px 8px 0 0;
    background-color: transparent !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    color: #f8fafc !important;
    border-bottom: 2px solid var(--accent-indigo) !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. MODEL & DATA CACHED LOADERS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "Exam_Score_Prediction.csv")

FEATURE_COLS = [
    'gender', 'course', 'study_hours', 'class_attendance',
    'internet_access', 'sleep_hours', 'sleep_quality',
    'study_method', 'facility_rating', 'exam_difficulty'
]

@st.cache_resource(show_spinner=False)
def load_trained_model():
    """Loads the pre-trained scikit-learn Pipeline (preprocessing + XGBoost)."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        pipeline = pickle.load(f)
    return pipeline

@st.cache_data(show_spinner=False)
def load_dataset_and_metrics():
    """
    Loads Exam_Score_Prediction.csv and computes real holdout test metrics
    using the exact split (test_size=0.2, random_state=42) from Boosting.ipynb.
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    clean_df = df.drop(columns=['student_id', 'age'], errors='ignore')

    # Model evaluation metrics verified from notebook
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    X = clean_df[FEATURE_COLS]
    y = clean_df['exam_score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = load_trained_model()
    y_pred_test = model.predict(X_test)

    test_metrics = {
        'r2': round(float(r2_score(y_test, y_pred_test)), 4),
        'mae': round(float(mean_absolute_error(y_test, y_pred_test)), 4),
        'rmse': round(float(np.sqrt(mean_squared_error(y_test, y_pred_test))), 4),
        'test_samples': len(y_test),
        'total_samples': len(df),
        'score_mean': round(float(df['exam_score'].mean()), 2),
        'score_std': round(float(df['exam_score'].std()), 2),
        'score_min': round(float(df['exam_score'].min()), 1),
        'score_max': round(float(df['exam_score'].max()), 1),
    }

    # Extract feature importances directly from active XGBoost pipeline
    xgb_step = model.named_steps.get('model')
    prep_step = model.named_steps.get('preprocessor')
    
    try:
        raw_names = prep_step.get_feature_names_out()
    except Exception:
        raw_names = [f"f_{i}" for i in range(len(xgb_step.feature_importances_))]
    
    importances = xgb_step.feature_importances_
    
    # Clean feature names for clean UI display
    fi_list = []
    for name, imp in zip(raw_names, importances):
        clean_name = (
            name.replace("cat_ord__", "")
                .replace("cat_one__", "")
                .replace("remainder__", "")
                .replace("_", " ")
                .title()
        )
        fi_list.append({"feature": clean_name, "importance": round(float(imp), 4)})
    
    fi_df = pd.DataFrame(fi_list).sort_values("importance", ascending=False).reset_index(drop=True)

    # Valid categorical values from dataset
    cat_options = {
        'gender': sorted(df['gender'].unique().tolist()),
        'course': sorted(df['course'].unique().tolist()),
        'internet_access': sorted(df['internet_access'].unique().tolist()),
        'sleep_quality': ['poor', 'average', 'good'],
        'study_method': sorted(df['study_method'].unique().tolist()),
        'facility_rating': ['low', 'medium', 'high'],
        'exam_difficulty': ['easy', 'moderate', 'hard']
    }

    return df, test_metrics, fi_df, cat_options


# Initialize data & model
try:
    pipeline_model = load_trained_model()
    raw_df, metrics, fi_dataframe, categorical_options = load_dataset_and_metrics()
except Exception as e:
    st.error(f"Error loading model or dataset: {e}")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# 4. HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-badge">
    <span class="dot"></span>
    XGBoost Pipeline · Live Model
</div>
<h1 class="brand-title">EXAMIQ</h1>
<div class="brand-subtitle">Exam Score Intelligence</div>
<p class="brand-desc">Predict outcomes. Understand performance.</p>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 5. PERFORMANCE OVERVIEW (GENUINE EVALUATION METRICS)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="metrics-grid">
    <div class="metric-box">
        <div class="metric-title">Active Model</div>
        <div class="metric-num" style="font-size: 1.25rem; font-family: 'Plus Jakarta Sans'; font-weight: 800; color: #a5b4fc;">
            Tuned XGBoost
        </div>
        <div class="metric-sub">Pipeline with ColumnTransformer</div>
    </div>
    <div class="metric-box">
        <div class="metric-title">Test R² Score</div>
        <div class="metric-num" style="color: #38bdf8;">{metrics['r2']}</div>
        <div class="metric-sub">Holdout Set (4,000 students)</div>
    </div>
    <div class="metric-box">
        <div class="metric-title">Mean Absolute Error</div>
        <div class="metric-num" style="color: #818cf8;">{metrics['mae']}</div>
        <div class="metric-sub">Average error in score points</div>
    </div>
    <div class="metric-box">
        <div class="metric-title">Root Mean Squared Error</div>
        <div class="metric-num" style="color: #c084fc;">{metrics['rmse']}</div>
        <div class="metric-sub">CV Best Score: 9.864</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 6. PREDICT EXAM SCORE (INTERACTIVE PREDICTION FLOW)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="glass-panel">
    <div class="section-title">
        <span class="icon">🎯</span> Predict Exam Score
    </div>
    <div class="section-desc">
        Configure student attributes and behavioral factors. Features are passed directly to the trained XGBoost pipeline.
    </div>
""", unsafe_allow_html=True)

# Presets Container Bar
st.markdown("""
<div class="presets-bar">
    <div class="presets-header">
        <span class="presets-title">⚡ Quick Profiles</span>
        <span class="presets-hint">Pre-fill realistic student profiles with 1 click</span>
    </div>
</div>
""", unsafe_allow_html=True)

# 3 equal columns for presets — NO cramped wrapping
col_p1, col_p2, col_p3 = st.columns(3, gap="medium")

if "form_defaults" not in st.session_state:
    st.session_state.form_defaults = {
        'gender': 'male',
        'course': 'b.tech',
        'study_hours': 4.0,
        'class_attendance': 75.0,
        'internet_access': 'yes',
        'sleep_hours': 7.0,
        'sleep_quality': 'average',
        'study_method': 'self-study',
        'facility_rating': 'medium',
        'exam_difficulty': 'moderate'
    }

with col_p1:
    if st.button("⚡ High Achiever", key="preset_high", type="secondary", use_container_width=True):
        st.session_state.form_defaults = {
            'gender': 'female',
            'course': 'b.tech',
            'study_hours': 7.2,
            'class_attendance': 95.0,
            'internet_access': 'yes',
            'sleep_hours': 7.5,
            'sleep_quality': 'good',
            'study_method': 'coaching',
            'facility_rating': 'high',
            'exam_difficulty': 'moderate'
        }
        st.rerun()

with col_p2:
    if st.button("🎯 Balanced Student", key="preset_med", type="secondary", use_container_width=True):
        st.session_state.form_defaults = {
            'gender': 'male',
            'course': 'bca',
            'study_hours': 4.5,
            'class_attendance': 78.0,
            'internet_access': 'yes',
            'sleep_hours': 7.0,
            'sleep_quality': 'average',
            'study_method': 'mixed',
            'facility_rating': 'medium',
            'exam_difficulty': 'moderate'
        }
        st.rerun()

with col_p3:
    if st.button("🔄 Reset Defaults", key="preset_reset", type="secondary", use_container_width=True):
        st.session_state.form_defaults = {
            'gender': 'male',
            'course': 'b.tech',
            'study_hours': 4.0,
            'class_attendance': 75.0,
            'internet_access': 'yes',
            'sleep_hours': 7.0,
            'sleep_quality': 'average',
            'study_method': 'self-study',
            'facility_rating': 'medium',
            'exam_difficulty': 'moderate'
        }
        st.rerun()

st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

# Current form state
curr = st.session_state.form_defaults

input_col1, input_col2 = st.columns(2, gap="large")

with input_col1:
    st.markdown("<p style='font-size:0.78rem; font-weight:700; color:#818cf8; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:12px;'>Academic & Study Factors</p>", unsafe_allow_html=True)
    
    course = st.selectbox(
        "Course Program",
        options=categorical_options['course'],
        index=categorical_options['course'].index(curr['course']) if curr['course'] in categorical_options['course'] else 0,
        format_func=lambda x: x.upper() if x in ['bca', 'bba', 'ba'] else x.title()
    )
    
    study_hours = st.slider(
        "Daily Study Hours",
        min_value=0.1,
        max_value=8.0,
        value=float(curr['study_hours']),
        step=0.1,
        help="Self-study and revision hours spent per day"
    )
    
    class_attendance = st.slider(
        "Class Attendance (%)",
        min_value=40.0,
        max_value=100.0,
        value=float(curr['class_attendance']),
        step=0.5,
        help="Lecture and seminar attendance rate"
    )
    
    study_method = st.selectbox(
        "Primary Study Method",
        options=categorical_options['study_method'],
        index=categorical_options['study_method'].index(curr['study_method']) if curr['study_method'] in categorical_options['study_method'] else 0,
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    exam_difficulty = st.selectbox(
        "Anticipated Exam Difficulty",
        options=categorical_options['exam_difficulty'],
        index=categorical_options['exam_difficulty'].index(curr['exam_difficulty']) if curr['exam_difficulty'] in categorical_options['exam_difficulty'] else 1,
        format_func=lambda x: x.title()
    )

with input_col2:
    st.markdown("<p style='font-size:0.78rem; font-weight:700; color:#818cf8; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:12px;'>Personal & Campus Environment</p>", unsafe_allow_html=True)
    
    gender = st.selectbox(
        "Gender Identity",
        options=categorical_options['gender'],
        index=categorical_options['gender'].index(curr['gender']) if curr['gender'] in categorical_options['gender'] else 0,
        format_func=lambda x: x.title()
    )
    
    sleep_hours = st.slider(
        "Nightly Sleep (Hours)",
        min_value=4.0,
        max_value=10.0,
        value=float(curr['sleep_hours']),
        step=0.1,
        help="Average hours of sleep per night"
    )
    
    sleep_quality = st.selectbox(
        "Sleep Quality",
        options=categorical_options['sleep_quality'],
        index=categorical_options['sleep_quality'].index(curr['sleep_quality']) if curr['sleep_quality'] in categorical_options['sleep_quality'] else 1,
        format_func=lambda x: x.title()
    )
    
    internet_access = st.selectbox(
        "High-Speed Internet Access",
        options=categorical_options['internet_access'],
        index=categorical_options['internet_access'].index(curr['internet_access']) if curr['internet_access'] in categorical_options['internet_access'] else 0,
        format_func=lambda x: "Yes (Available)" if x == 'yes' else "No (Limited)"
    )
    
    facility_rating = st.selectbox(
        "Campus & Library Facility Rating",
        options=categorical_options['facility_rating'],
        index=categorical_options['facility_rating'].index(curr['facility_rating']) if curr['facility_rating'] in categorical_options['facility_rating'] else 1,
        format_func=lambda x: x.title()
    )

st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

# Prediction Action with primary accent
predict_clicked = st.button("⚡ Calculate Predicted Score", key="btn_calculate", type="primary", use_container_width=True)

if predict_clicked:
    try:
        # Construct single-row DataFrame with the exact feature names and order
        input_data = pd.DataFrame([{
            'gender': gender,
            'course': course,
            'study_hours': float(study_hours),
            'class_attendance': float(class_attendance),
            'internet_access': internet_access,
            'sleep_hours': float(sleep_hours),
            'sleep_quality': sleep_quality,
            'study_method': study_method,
            'facility_rating': facility_rating,
            'exam_difficulty': exam_difficulty
        }])

        with st.spinner("Processing input through XGBoost pipeline..."):
            pred_score = float(pipeline_model.predict(input_data)[0])
            pred_score = float(np.clip(pred_score, 0.0, 100.0))

        # Percentile rank against the real 20,000 dataset
        all_scores = raw_df['exam_score'].values
        percentile = float(np.mean(all_scores <= pred_score) * 100.0)

        # Grade calculation
        if pred_score >= 90:
            grade, grade_col = "A+", "#10b981"
        elif pred_score >= 80:
            grade, grade_col = "A", "#3b82f6"
        elif pred_score >= 70:
            grade, grade_col = "B", "#8b5cf6"
        elif pred_score >= 60:
            grade, grade_col = "C", "#f59e0b"
        elif pred_score >= 50:
            grade, grade_col = "D", "#f97316"
        else:
            grade, grade_col = "F", "#ef4444"

        diff_from_mean = pred_score - metrics['score_mean']
        diff_str = f"+{diff_from_mean:.1f}" if diff_from_mean >= 0 else f"{diff_from_mean:.1f}"

        # Study impact appraisal
        study_tier = "High (>6 hrs)" if study_hours >= 6.0 else ("Moderate (3-6 hrs)" if study_hours >= 3.0 else "Low (<3 hrs)")
        att_tier = "Optimal (≥80%)" if class_attendance >= 80.0 else ("Good (70-80%)" if class_attendance >= 70.0 else "Needs Boost (<70%)")
        sleep_tier = f"{sleep_quality.title()} ({sleep_hours:.1f}h)"

        # Display result card
        st.markdown(f"""
        <div class="result-card">
            <span class="grade-badge" style="background: {grade_col}22; color: {grade_col}; border: 1px solid {grade_col}55;">
                Grade {grade}
            </span>
            <div class="result-score">{pred_score:.1f} <span style="font-size: 1.5rem; color: #64748b; font-weight: 500;">/ 100</span></div>
            <div style="font-size: 1.05rem; font-weight: 600; color: #f1f5f9; margin-bottom: 4px;">
                {percentile:.1f}th Cohort Percentile
            </div>
            <div style="font-size: 0.86rem; color: #94a3b8;">
                {diff_str} points relative to dataset cohort average ({metrics['score_mean']})
            </div>
            <div class="factors-summary">
                <div class="factor-chip">
                    <div class="factor-chip-title">Study Habit</div>
                    <div class="factor-chip-val">{study_tier}</div>
                </div>
                <div class="factor-chip">
                    <div class="factor-chip-title">Attendance</div>
                    <div class="factor-chip-val">{att_tier}</div>
                </div>
                <div class="factor-chip">
                    <div class="factor-chip-title">Rest Cycle</div>
                    <div class="factor-chip-val">{sleep_tier}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Gauge visualization
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(pred_score, 1),
            domain={'x': [0, 1], 'y': [0, 1]},
            number={'suffix': " pts", 'font': {'color': '#f8fafc', 'family': 'Plus Jakarta Sans', 'size': 30}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#334155', 'tickfont': {'color': '#94a3b8', 'size': 11}},
                'bar': {'color': grade_col, 'thickness': 0.28},
                'bgcolor': "rgba(255, 255, 255, 0.04)",
                'borderwidth': 1,
                'bordercolor': "rgba(255, 255, 255, 0.08)",
                'steps': [
                    {'range': [0, 50], 'color': "rgba(239, 68, 68, 0.12)"},
                    {'range': [50, 70], 'color': "rgba(245, 158, 11, 0.12)"},
                    {'range': [70, 85], 'color': "rgba(99, 102, 241, 0.12)"},
                    {'range': [85, 100], 'color': "rgba(16, 185, 129, 0.12)"}
                ],
                'threshold': {
                    'line': {'color': "#ffffff", 'width': 3},
                    'thickness': 0.8,
                    'value': round(metrics['score_mean'], 1)
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=195,
            margin=dict(l=20, r=20, t=15, b=10)
        )
        st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})

    except Exception as err:
        st.error(f"Prediction failed: {err}")

st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 7. MODEL PERFORMANCE & ENSEMBLE BENCHMARKS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="glass-panel">
    <div class="section-title">
        <span class="icon">📊</span> Model Performance & Intelligence
    </div>
    <div class="section-desc">
        Genuine evaluation results directly derived from Boosting.ipynb and the trained pipeline model.
    </div>
""", unsafe_allow_html=True)

tab_comp, tab_fi, tab_dist = st.tabs([
    "🏆 Model Leaderboard", 
    "🔍 Feature Importances", 
    "📈 Dataset Distribution"
])

# ── Tab 1: Leaderboard from Notebook ──
with tab_comp:
    st.markdown("<p style='font-size:0.86rem; color:#94a3b8; margin-bottom:14px;'>Comparative evaluation of all regression algorithms trained in <code>Boosting.ipynb</code> on holdout test data:</p>", unsafe_allow_html=True)
    
    # Exact genuine results extracted from Boosting.ipynb
    model_benchmarks = pd.DataFrame([
        {"Model": "Decision Tree Regressor", "R2": 0.3945, "MAE": 12.45, "RMSE": 15.62, "Type": "Baseline"},
        {"Model": "AdaBoost (Exponential)", "R2": 0.6797, "MAE": 8.7327, "RMSE": 10.7040, "Type": "Ensemble"},
        {"Model": "AdaBoost (Linear)", "R2": 0.6806, "MAE": 8.7105, "RMSE": 10.6891, "Type": "Ensemble"},
        {"Model": "AdaBoost (Square)", "R2": 0.6883, "MAE": 8.5873, "RMSE": 10.5591, "Type": "Ensemble"},
        {"Model": "XGBoost (Untuned)", "R2": 0.7002, "MAE": 8.2964, "RMSE": 10.3562, "Type": "Gradient Boosted"},
        {"Model": "GradientBoosting (Untuned)", "R2": 0.7290, "MAE": 7.9404, "RMSE": 9.8457, "Type": "Gradient Boosted"},
        {"Model": "GradientBoosting (Tuned)", "R2": 0.7304, "MAE": 7.9010, "RMSE": 9.8194, "Type": "Gradient Boosted"},
        {"Model": "XGBoost Pipeline (Active)", "R2": 0.7305, "MAE": 7.9223, "RMSE": 9.8185, "Type": "Production (Saved)"},
    ]).sort_values("R2", ascending=True)

    fig_comp = go.Figure()

    colors = [
        '#6366f1' if "Active" in m else ('#818cf8' if "Gradient" in m else '#475569') 
        for m in model_benchmarks['Model']
    ]

    fig_comp.add_trace(go.Bar(
        y=model_benchmarks['Model'],
        x=model_benchmarks['R2'],
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(255, 255, 255, 0.15)', width=1)
        ),
        text=[f"R² = {r:.4f}" for r in model_benchmarks['R2']],
        textposition='outside',
        textfont=dict(color='#cbd5e1', family='JetBrains Mono', size=11),
        hoverinfo='text',
        hovertext=[f"<b>{row['Model']}</b><br>R² Score: {row['R2']:.4f}<br>MAE: {row['MAE']:.2f}<br>RMSE: {row['RMSE']:.2f}" for _, row in model_benchmarks.iterrows()]
    ))

    fig_comp.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=380,
        margin=dict(l=10, r=60, t=10, b=20),
        xaxis=dict(
            title="Coefficient of Determination (R² Score)",
            title_font=dict(color='#94a3b8', size=12),
            tickfont=dict(color='#64748b'),
            gridcolor='rgba(255, 255, 255, 0.05)',
            range=[0.3, 0.8]
        ),
        yaxis=dict(
            tickfont=dict(color='#f1f5f9', size=12),
        )
    )
    st.plotly_chart(fig_comp, use_container_width=True, config={'displayModeBar': False})

# ── Tab 2: Feature Importances from Pipeline ──
with tab_fi:
    st.markdown("<p style='font-size:0.86rem; color:#94a3b8; margin-bottom:14px;'>Feature contributions computed directly from the saved XGBoost regressor inside <code>model.pkl</code>:</p>", unsafe_allow_html=True)
    
    top_fi = fi_dataframe.head(10).sort_values("importance", ascending=True)
    
    fig_fi = go.Figure(go.Bar(
        y=top_fi['feature'],
        x=top_fi['importance'],
        orientation='h',
        marker=dict(
            color='rgba(99, 102, 241, 0.85)',
            line=dict(color='rgba(165, 180, 252, 0.4)', width=1)
        ),
        text=[f"{v*100:.1f}%" for v in top_fi['importance']],
        textposition='outside',
        textfont=dict(color='#cbd5e1', family='JetBrains Mono', size=11),
        hovertemplate="<b>%{y}</b><br>Importance Weight: %{x:.4f}<extra></extra>"
    ))
    fig_fi.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=360,
        margin=dict(l=10, r=50, t=10, b=20),
        xaxis=dict(
            title="Relative Importance Ratio",
            title_font=dict(color='#94a3b8', size=12),
            tickfont=dict(color='#64748b'),
            gridcolor='rgba(255, 255, 255, 0.05)',
            range=[0, top_fi['importance'].max() * 1.2]
        ),
        yaxis=dict(
            tickfont=dict(color='#f1f5f9', size=12),
        )
    )
    st.plotly_chart(fig_fi, use_container_width=True, config={'displayModeBar': False})

# ── Tab 3: Dataset Distribution ──
with tab_dist:
    st.markdown(f"<p style='font-size:0.86rem; color:#94a3b8; margin-bottom:14px;'>Score distribution across all {metrics['total_samples']:,} students in <code>Exam_Score_Prediction.csv</code> (Mean: {metrics['score_mean']}, Std: {metrics['score_std']}):</p>", unsafe_allow_html=True)
    
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=raw_df['exam_score'],
        nbinsx=35,
        marker=dict(
            color='rgba(139, 92, 246, 0.55)',
            line=dict(color='rgba(168, 85, 247, 0.8)', width=1)
        ),
        hovertemplate="Score Range: %{x}<br>Students: %{y}<extra></extra>"
    ))
    fig_dist.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=320,
        margin=dict(l=10, r=20, t=10, b=20),
        xaxis=dict(
            title="Exam Score (0 - 100)",
            title_font=dict(color='#94a3b8', size=12),
            tickfont=dict(color='#64748b'),
            gridcolor='rgba(255, 255, 255, 0.05)'
        ),
        yaxis=dict(
            title="Frequency",
            title_font=dict(color='#94a3b8', size=12),
            tickfont=dict(color='#64748b'),
            gridcolor='rgba(255, 255, 255, 0.05)'
        )
    )
    st.plotly_chart(fig_dist, use_container_width=True, config={'displayModeBar': False})

st.markdown("</div>", unsafe_allow_html=True)
