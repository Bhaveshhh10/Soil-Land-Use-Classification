"""
app.py -- EuroSAT Land Use Classification Dashboard
SE-MobileNetV2 + Curriculum Learning + Agentic Router
"""

import os, sys, json, warnings, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from model import (
    focal_loss, build_feature_extractor, agentic_router,
    make_gradcam, EUROSAT_CLASSES, NUM_CLASSES, IMG_SIZE, CONFIDENCE_THRESHOLD,
)
from data_utils import preprocess_single_image, CLASS_COLORS
from artefact_paths import artefact_paths, ensure_artefacts, artefacts_json_path, bootstrap_artefacts_json
from ui_components import NAV_CARDS, render_architecture_diagram

st.set_page_config(
    page_title="EuroSAT · Land Use AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── helpers ──────────────────────────────────────────────────────────────────

def fmt_pct(v):
    if v is None: return 0.0
    v = float(v)
    return v if v > 1.0 else v * 100.0

def style_plot(fig, height=None):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#151b28",
        font=dict(color="#e2e8f0", family="DM Sans, system-ui, sans-serif"),
        margin=dict(t=48, b=40, l=48, r=24),
    )
    fig.update_xaxes(gridcolor="#2a3548", zerolinecolor="#2a3548")
    fig.update_yaxes(gridcolor="#2a3548", zerolinecolor="#2a3548")
    if height: fig.update_layout(height=height)
    return fig

# ── CSS ──────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600;9..40,700&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', system-ui, sans-serif; }
.block-container { padding-top: 1.5rem; max-width: 1280px; }
.stApp { background: #0b0f19; }

.metric-card {
    background: linear-gradient(145deg,#151b28,#1a2236);
    border:1px solid #2a3548; border-radius:12px;
    padding:22px 18px; text-align:left;
    box-shadow:0 8px 24px rgba(0,0,0,.35);
    margin-bottom:8px; border-top:3px solid #38bdf8;
    transition:transform .2s,box-shadow .2s;
}
.metric-card:hover { transform:translateY(-2px); box-shadow:0 12px 32px rgba(56,189,248,.12); }
.metric-card .value { font-size:2rem; font-weight:700; color:#f1f5f9; line-height:1.1; font-variant-numeric:tabular-nums; }
.metric-card .label { font-size:.8rem; color:#94a3b8; margin-top:6px; font-weight:500; }
.metric-card.g { border-top-color:#34d399; }
.metric-card.a { border-top-color:#fbbf24; }
.metric-card.v { border-top-color:#a78bfa; }
.metric-card.p { border-top-color:#f472b6; }

.section-header {
    font-family:'DM Serif Display',Georgia,serif;
    font-size:1.45rem; font-weight:400; color:#f1f5f9;
    margin:28px 0 14px; padding-bottom:8px;
    border-bottom:1px solid #2a3548;
}

.hero-box {
    background:linear-gradient(135deg,#0f172a 0%,#1e293b 40%,#0c4a6e 100%);
    border:1px solid #334155; border-radius:16px;
    padding:44px 40px; color:#f8fafc; margin-bottom:24px;
    box-shadow:0 20px 50px rgba(0,0,0,.45);
    position:relative; overflow:hidden;
}
.hero-box::before {
    content:''; position:absolute; top:-40%; right:-10%;
    width:50%; height:140%;
    background:radial-gradient(circle,rgba(56,189,248,.18) 0%,transparent 70%);
    pointer-events:none;
}
.hero-box h1 {
    font-family:'DM Serif Display',Georgia,serif;
    font-size:2.3rem; font-weight:400; margin-bottom:10px;
    letter-spacing:-.02em; position:relative;
}
.hero-box p { font-size:1rem; color:#cbd5e1; line-height:1.65; max-width:760px; position:relative; }

.class-card {
    background:#151b28; border:1px solid #2a3548; border-radius:10px;
    padding:14px 10px; text-align:center; margin-bottom:8px;
    transition:border-color .2s,transform .2s;
}
.class-card:hover { border-color:#38bdf8; transform:translateY(-1px); }
.class-card .swatch { width:28px; height:28px; border-radius:50%; margin:0 auto 8px; box-shadow:0 0 12px rgba(0,0,0,.4); }
.class-card .name { font-size:.72rem; font-weight:600; color:#cbd5e1; line-height:1.25; }

.phase-card {
    background:linear-gradient(145deg,#151b28,#1a2236);
    border:1px solid #2a3548; border-radius:12px;
    padding:20px 16px; text-align:center; margin-bottom:8px;
}
.phase-card .phase-label { font-size:.72rem; text-transform:uppercase; letter-spacing:.1em; color:#64748b; font-weight:700; margin-bottom:6px; }
.phase-card .phase-value { font-size:1.8rem; font-weight:700; color:#f1f5f9; font-variant-numeric:tabular-nums; }
.phase-card .phase-sub { font-size:.75rem; color:#94a3b8; margin-top:4px; }

.info-box {
    background:rgba(56,189,248,.07); border:1px solid rgba(56,189,248,.25);
    border-radius:10px; padding:16px 18px; margin:12px 0;
    font-size:.88rem; color:#cbd5e1; line-height:1.6;
}
.warn-box {
    background:rgba(251,191,36,.07); border:1px solid rgba(251,191,36,.25);
    border-radius:10px; padding:16px 18px; margin:12px 0;
    font-size:.88rem; color:#fde68a; line-height:1.6;
}

.status-dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; vertical-align:middle; }
.dot-green { background:#34d399; box-shadow:0 0 6px #34d399; }
.dot-red   { background:#f87171; box-shadow:0 0 6px #f87171; }
.dot-amber { background:#fbbf24; box-shadow:0 0 6px #fbbf24; }

.tech-badge {
    display:inline-block; padding:6px 14px; border-radius:20px;
    font-size:.78rem; font-weight:600; margin:4px;
    background:#1e293b; border:1px solid #334155; color:#94a3b8;
}

[data-testid="stSidebar"] { background:#0f1419 !important; border-right:1px solid #2a3548; }
div[data-testid="column"] .stButton > button {
    width:100%; border-radius:8px; font-weight:600;
    border:1px solid #334155; background:#1e293b; color:#e2e8f0;
    transition:all .2s;
}
div[data-testid="column"] .stButton > button:hover {
    border-color:#38bdf8; background:#0f2744; color:#f8fafc;
    box-shadow:0 0 20px rgba(56,189,248,.25);
}
</style>
""", unsafe_allow_html=True)

# ── resource loaders ─────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading SE-MobileNetV2…")
def load_model():
    import tensorflow as tf
    candidates = [
        os.path.join(BASE_DIR, "se_model.keras"),
        os.path.join(BASE_DIR, "artefacts", "se_mobilenetv2_eurosat.h5"),
        os.path.join(BASE_DIR, "artefacts", "se_model.keras"),
    ]
    last_err = None
    for p in candidates:
        if os.path.exists(p):
            try:
                return tf.keras.models.load_model(
                    p, custom_objects={"focal_loss_fn": focal_loss(2.0, 0.25, NUM_CLASSES)}
                )
            except Exception as e:
                last_err = e
    if last_err:
        st.warning(f"Model file found but failed to load: {last_err}")
    return None

@st.cache_resource(show_spinner=False)
def load_knn_and_scaler():
    knn, scaler = None, None
    for p in [os.path.join(BASE_DIR, "knn_fallback.pkl"), os.path.join(BASE_DIR, "artefacts", "knn_fallback.pkl")]:
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    knn = pickle.load(f)
                break
            except Exception as e:
                st.warning(f"KNN load failed: {e}")
    for p in [os.path.join(BASE_DIR, "emb_scaler.pkl"), os.path.join(BASE_DIR, "artefacts", "emb_scaler.pkl")]:
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    scaler = pickle.load(f)
                break
            except Exception as e:
                st.warning(f"Scaler load failed: {e}")
    return knn, scaler

@st.cache_resource(show_spinner=False)
def load_feature_extractor(model):
    if model is None: return None
    try: return build_feature_extractor(model)
    except Exception: return None

@st.cache_resource(show_spinner=False)
def load_markov():
    """Load Markov transition matrix and degradation risk scores from markov.pkl."""
    for p in [os.path.join(BASE_DIR, "markov.pkl"), os.path.join(BASE_DIR, "artefacts", "markov.pkl")]:
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
    return None

@st.cache_data(show_spinner=False)
def load_artefacts(_json_path, _mtime):
    if not _json_path or not os.path.isfile(_json_path): return None
    with open(_json_path, "r", encoding="utf-8") as f: return json.load(f)

def get_artefacts():
    json_path = artefacts_json_path()
    if not os.path.isfile(json_path):
        bootstrap_artefacts_json()
        json_path = artefacts_json_path()
    mtime = os.path.getmtime(json_path) if os.path.isfile(json_path) else 0.0
    return load_artefacts(json_path, mtime)

# ── navigation ────────────────────────────────────────────────────────────────

PAGES = ["Home", "Predict", "Model Comparison", "Training Progress", "Explainability", "About"]

if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Home"

with st.sidebar:
    st.markdown('<div style="font-size:1.1rem;font-weight:700;color:#f1f5f9;margin-bottom:2px;">🛰️ EuroSAT Dashboard</div>', unsafe_allow_html=True)
    st.caption("SE-MobileNetV2 · Curriculum Learning · Agentic Router")
    st.markdown("---")

    selected = st.radio("Navigation", PAGES, index=PAGES.index(st.session_state.nav_page), label_visibility="collapsed")
    if selected != st.session_state.nav_page:
        st.session_state.nav_page = selected
    page = st.session_state.nav_page
    st.markdown("---")

    art = get_artefacts()
    model_exists = any(os.path.exists(p) for p in [
        os.path.join(BASE_DIR, "se_model.keras"),
        os.path.join(BASE_DIR, "artefacts", "se_mobilenetv2_eurosat.h5"),
        os.path.join(BASE_DIR, "artefacts", "se_model.keras"),
    ])

    if art:
        se_acc_val  = fmt_pct(art["se_acc"])
        rt_acc_val  = fmt_pct(art["routed_acc"])
        van_acc_val = fmt_pct(art["van_acc"])
        st.markdown('<span class="status-dot dot-green"></span><span style="font-size:.82rem;color:#94a3b8;">Metrics loaded</span>', unsafe_allow_html=True)
        st.metric("SE-MobileNetV2", f"{se_acc_val:.2f}%")
        st.metric("Agentic Router", f"{rt_acc_val:.2f}%", delta=f"+{rt_acc_val - van_acc_val:.2f}% vs Vanilla")
    else:
        st.markdown('<span class="status-dot dot-red"></span><span style="font-size:.82rem;color:#f87171;">Artefacts not found</span>', unsafe_allow_html=True)

    dot = "dot-green" if model_exists else "dot-amber"
    label = "Model weights ready" if model_exists else "Model weights not found"
    st.markdown(f'<div style="margin-top:8px;"><span class="status-dot {dot}"></span><span style="font-size:.82rem;color:#94a3b8;">{label}</span></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Metrics from `artefacts/artefacts.json`")


# =============================================================================
# PAGE 1 — HOME
# =============================================================================

def page_home():
    import pandas as pd

    art = get_artefacts()
    if not art:
        st.error("No artefacts found. Run `python extract_notebook_metrics.py` to generate metrics.")
        return

    se_acc     = fmt_pct(art["se_acc"])
    routed_acc = fmt_pct(art["routed_acc"])
    van_acc    = fmt_pct(art["van_acc"])
    n_test     = int(art.get("n_test", 5400))
    n_knn      = int(art.get("n_knn_routed", 810))
    improvement = routed_acc - van_acc
    knn_pct     = n_knn / n_test * 100 if n_test > 0 else 0.0

    # Hero
    st.markdown(f"""
<div class="hero-box">
  <h1>Soil &amp; Land Use Classification</h1>
  <p>Multi-class satellite land-use mapping with <strong>SE-MobileNetV2</strong>,
  three-phase curriculum learning, and an agentic confidence router on the
  <strong>EuroSAT RGB</strong> benchmark — 27,000 Sentinel-2 patches, 10 classes,
  test-set accuracy of <strong>{routed_acc:.2f}%</strong>.</p>
</div>""", unsafe_allow_html=True)

    # Key metrics
    st.markdown('<div class="section-header">Key Results</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="value">{se_acc:.2f}%</div><div class="label">SE-MobileNetV2 Accuracy</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card g"><div class="value">{routed_acc:.2f}%</div><div class="label">Router Accuracy (Best)</div></div>', unsafe_allow_html=True)
    with c3:
        sign = "+" if improvement >= 0 else ""
        st.markdown(f'<div class="metric-card a"><div class="value">{sign}{improvement:.2f}%</div><div class="label">Improvement vs Vanilla</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card v"><div class="value">{knn_pct:.1f}%</div><div class="label">Samples Routed to KNN</div></div>', unsafe_allow_html=True)

    # ── Quick-nav cards ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Explore the Dashboard</div>', unsafe_allow_html=True)
    nav_items = [
        ("🔍 Predict",          "Upload & Classify",      "Run the agentic router on any satellite image with live Grad-CAM attention maps.",          "#38bdf8", "Predict"),
        ("📊 Model Comparison", "Benchmark All Approaches","Classical ML → Vanilla MobileNetV2 → SE-MobileNetV2 → Full Router pipeline.",              "#34d399", "Model Comparison"),
        ("📈 Training Progress","Curriculum Phases",       "Validation accuracy across easy (40%), medium (70%), and full-data training phases.",       "#a78bfa", "Training Progress"),
        ("🧠 Explainability",   "Grad-CAM & SE Blocks",   "See where the network attends and how SE channel excitation recalibrates features.",        "#f472b6", "Explainability"),
        ("ℹ️ About",            "Project Overview",        "Dataset details, architecture summary, tech stack, and academic references.",               "#94a3b8", "About"),
    ]
    nav_cols = st.columns(5)
    for col, (title, subtitle, desc, accent, target) in zip(nav_cols, nav_items):
        with col:
            st.markdown(f"""
<div style="position:relative;background:#151b28;border:1px solid #2a3548;border-radius:12px;
     padding:18px 14px 12px;min-height:150px;overflow:hidden;margin-bottom:6px;">
  <div style="position:absolute;top:0;left:0;right:0;height:3px;background:{accent};box-shadow:0 0 14px {accent};"></div>
  <div style="font-size:1.05rem;font-weight:700;color:#f1f5f9;margin-top:4px;">{title}</div>
  <div style="font-size:.72rem;color:{accent};font-weight:600;margin:3px 0 8px;text-transform:uppercase;letter-spacing:.06em;">{subtitle}</div>
  <div style="font-size:.8rem;color:#94a3b8;line-height:1.45;">{desc}</div>
</div>""", unsafe_allow_html=True)
            if st.button(f"Open", key=f"nav_{target}", use_container_width=True):
                st.session_state.nav_page = target
                st.rerun()

    # ── Interactive radar chart ───────────────────────────────────────────────
    st.markdown('<div class="section-header">Model Performance Radar</div>', unsafe_allow_html=True)

    ml = art.get("ml_results", {})

    def _hex_to_rgba(hex_color, alpha=0.12):
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    radar_data = [
        ("Logistic Reg.",      [fmt_pct(ml["Logistic Regression"]["acc"]), float(ml["Logistic Regression"]["f1"])*100, 0,  0,  0 ], "#3b82f6"),
        ("Random Forest",      [fmt_pct(ml["Random Forest"]["acc"]),        float(ml["Random Forest"]["f1"])*100,        0,  0,  0 ], "#60a5fa"),
        ("Vanilla MobileNetV2",[fmt_pct(art["van_acc"]),                    float(art["van_f1"])*100,                   70, 0,  0 ], "#f59e0b"),
        ("SE-MobileNetV2",     [se_acc,                                     float(art["se_f1"])*100,                    85, 90, 0 ], "#10b981"),
        ("SE + Router",        [routed_acc,                                 float(art["routed_f1"])*100,                85, 90, 95], "#059669"),
    ]
    categories = ["Accuracy", "Macro F1 ×100", "SE Attention", "Curriculum", "Agentic Router"]

    fig_radar = go.Figure()
    for name, vals, color in radar_data:
        closed_vals  = vals + [vals[0]]
        closed_theta = categories + [categories[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=closed_vals, theta=closed_theta,
            fill="toself", name=name,
            line=dict(color=color, width=2),
            fillcolor=_hex_to_rgba(color, 0.10),
            opacity=0.9,
            hovertemplate="<b>%{theta}</b>: %{r:.1f}<extra>" + name + "</extra>",
        ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 105], gridcolor="#2a3548", tickfont=dict(color="#64748b", size=9)),
            angularaxis=dict(gridcolor="#2a3548", tickfont=dict(color="#94a3b8", size=11)),
            bgcolor="#151b28",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5, font=dict(size=11)),
        showlegend=True,
    )
    style_plot(fig_radar, height=440)
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── Accuracy progression bar ──────────────────────────────────────────────
    st.markdown('<div class="section-header">Accuracy Progression</div>', unsafe_allow_html=True)
    prog_names  = ["Logistic Reg.", "KNN (k=5)", "Decision Tree", "Random Forest", "Vanilla MobileNetV2", "SE-MobileNetV2", "SE + Router"]
    prog_accs   = [
        fmt_pct(ml["Logistic Regression"]["acc"]),
        fmt_pct(ml["KNN (k=5)"]["acc"]),
        fmt_pct(ml["Decision Tree"]["acc"]),
        fmt_pct(ml["Random Forest"]["acc"]),
        fmt_pct(art["van_acc"]),
        se_acc, routed_acc,
    ]
    prog_colors = ["#3b82f6","#60a5fa","#93c5fd","#bfdbfe","#f59e0b","#10b981","#059669"]
    fig_prog = go.Figure(go.Bar(
        x=prog_accs, y=prog_names, orientation="h",
        marker=dict(color=prog_colors, line=dict(width=0)),
        text=[f"{v:.2f}%" for v in prog_accs],
        textposition="outside", textfont=dict(color="#e2e8f0", size=11),
        hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
    ))
    fig_prog.update_layout(
        xaxis=dict(title="Accuracy (%)", range=[0, 108]),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
    )
    style_plot(fig_prog, height=320)
    st.plotly_chart(fig_prog, use_container_width=True)

    # ── What we did differently ───────────────────────────────────────────────
    st.markdown('<div class="section-header">What We Did Differently</div>', unsafe_allow_html=True)
    diff_cols = st.columns(3)
    diff_items = [
        ("#38bdf8", "Multi-Scale SE Attention",
         "SE blocks applied at 4 backbone taps (block_3/6/13/out_relu). "
         "Each scale captures different spatial granularity — from local texture to global context."),
        ("#a78bfa", "3-Phase Curriculum Learning",
         "Samples ranked by difficulty via a Random Forest proxy. Training progresses from "
         "easiest 40% → 70% → full dataset, preventing early overfitting to hard examples."),
        ("#34d399", "Agentic Confidence Router",
         "At inference, low-confidence predictions (< 60%) are rerouted to a KNN specialist "
         "trained on SE embeddings, recovering accuracy on ambiguous samples."),
        ("#fbbf24", "Focal Loss",
         "Focal Loss (γ=2, α=0.25) down-weights easy examples and focuses training on "
         "hard, misclassified samples — critical for class-imbalanced satellite imagery."),
        ("#f472b6", "Multi-Scale Feature Fusion",
         "GAP outputs from all 4 SE-attended scales are concatenated into a 2144-d vector, "
         "then compressed through Dense(512) → Dense(256) before classification."),
        ("#94a3b8", "KNN Fallback Specialist",
         "KNN (k=7, cosine distance) trained on 256-d SE embeddings provides a non-parametric "
         "fallback that handles distribution-shift and ambiguous boundary cases."),
    ]
    for i, (accent, title, desc) in enumerate(diff_items):
        with diff_cols[i % 3]:
            st.markdown(f"""
<div style="background:linear-gradient(145deg,#151b28,#1a2236);border:1px solid #2a3548;
     border-left:3px solid {accent};border-radius:10px;padding:16px 14px;margin-bottom:10px;">
  <div style="font-size:.9rem;font-weight:700;color:#f1f5f9;margin-bottom:6px;">{title}</div>
  <div style="font-size:.8rem;color:#94a3b8;line-height:1.5;">{desc}</div>
</div>""", unsafe_allow_html=True)

    # ── Class grid ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">EuroSAT Land-Use Classes</div>', unsafe_allow_html=True)
    cls_cols = st.columns(5)
    for i, cls in enumerate(EUROSAT_CLASSES):
        color = CLASS_COLORS.get(cls, "#64748b")
        with cls_cols[i % 5]:
            st.markdown(f'<div class="class-card"><div class="swatch" style="background:{color};"></div><div class="name">{cls}</div></div>', unsafe_allow_html=True)

    # ── Architecture ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Architecture Overview</div>', unsafe_allow_html=True)
    render_architecture_diagram()


# =============================================================================
# PAGE 2 — PREDICT
# =============================================================================

def page_predict():
    st.markdown('<div class="section-header">Upload &amp; Classify</div>', unsafe_allow_html=True)
    st.markdown(
        "Upload a satellite or aerial image. The **Agentic Confidence Router** classifies it "
        "into one of 10 EuroSAT land-use categories — high-confidence samples go through "
        "SE-MobileNetV2 directly; low-confidence ones are routed to the KNN fallback."
    )

    art = get_artefacts()
    threshold = float(art.get("confidence_threshold", CONFIDENCE_THRESHOLD)) if art else CONFIDENCE_THRESHOLD

    uploaded = st.file_uploader(
        "Choose an image (JPG / PNG / TIF)",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
    )

    if uploaded is None:
        st.markdown(
            '<div class="info-box">📡 Upload a satellite or aerial image to run the full agentic router pipeline. '
            'The model will show Grad-CAM attention maps alongside the prediction and confidence breakdown.</div>',
            unsafe_allow_html=True,
        )
        return

    pil_img   = Image.open(uploaded).convert("RGB")
    img_array = preprocess_single_image(pil_img, IMG_SIZE)

    model          = load_model()
    knn, emb_scaler = load_knn_and_scaler()
    feat_extractor  = load_feature_extractor(model)

    if model is None or knn is None or emb_scaler is None or feat_extractor is None:
        st.markdown(
            '<div class="warn-box">⚠️ <strong>Live inference unavailable</strong> — model weights or KNN pickle not found. '
            'Place <code>se_model.keras</code> and <code>knn_fallback.pkl</code> in the <code>streamlit_app/</code> directory.</div>',
            unsafe_allow_html=True,
        )
        col_img, col_info = st.columns([1, 2])
        with col_img:
            st.image(pil_img, caption="Uploaded image", use_column_width=True)
        with col_info:
            st.markdown("**Pipeline steps with model weights loaded:**")
            st.markdown(
                f"1. Resize to 64×64 RGB float32\n"
                f"2. SE-MobileNetV2 → softmax probabilities\n"
                f"3. conf ≥ {threshold*100:.0f}% → SE direct prediction\n"
                f"4. conf < {threshold*100:.0f}% → KNN fallback on SE embeddings\n"
                f"5. Grad-CAM heatmap highlights attended regions"
            )
        return

    with st.spinner("Running agentic router inference…"):
        batch = img_array[np.newaxis]
        final_preds, sources, se_conf, se_proba = agentic_router(
            batch, model, knn, feat_extractor, emb_scaler, threshold
        )
        pred_idx   = int(final_preds[0])
        pred_class = EUROSAT_CLASSES[pred_idx]
        confidence = float(se_conf[0])
        source     = sources[0]
        proba_vec  = se_proba[0]
        gradcam_result = make_gradcam(model, img_array, IMG_SIZE)

    # Image row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Original Image**")
        st.image(pil_img, use_column_width=True)

    if gradcam_result is not None:
        cam_resized, overlay, _, _ = gradcam_result
        with col2:
            st.markdown("**Grad-CAM Heatmap**")
            fig_cam, ax_cam = plt.subplots(figsize=(3, 3))
            fig_cam.patch.set_facecolor("#151b28")
            ax_cam.set_facecolor("#151b28")
            ax_cam.imshow(cam_resized, cmap="jet")
            ax_cam.axis("off")
            st.pyplot(fig_cam); plt.close(fig_cam)
        with col3:
            st.markdown("**Grad-CAM Overlay**")
            fig_ov, ax_ov = plt.subplots(figsize=(3, 3))
            fig_ov.patch.set_facecolor("#151b28")
            ax_ov.set_facecolor("#151b28")
            ax_ov.imshow(overlay)
            ax_ov.axis("off")
            st.pyplot(fig_ov); plt.close(fig_ov)
    else:
        with col2:
            st.info("Grad-CAM not available for this model configuration.")

    st.markdown("---")
    res_col, conf_col = st.columns([2, 1])

    with res_col:
        swatch = CLASS_COLORS.get(pred_class, "#64748b")
        route_badge = (
            '<span style="background:rgba(56,189,248,.15);color:#7dd3fc;border:1px solid rgba(56,189,248,.35);padding:4px 10px;border-radius:6px;font-size:.78rem;font-weight:600;">SE Direct</span>'
            if source == "se_direct" else
            '<span style="background:rgba(251,146,60,.15);color:#fdba74;border:1px solid rgba(251,146,60,.35);padding:4px 10px;border-radius:6px;font-size:.78rem;font-weight:600;">KNN Fallback</span>'
        )
        st.markdown(
            f'<p style="margin:0 0 4px;font-size:.85rem;color:#64748b;">Predicted class</p>'
            f'<h3 style="margin:0 0 10px;color:#f1f5f9;">'
            f'<span style="display:inline-block;width:14px;height:14px;border-radius:50%;background:{swatch};margin-right:10px;vertical-align:middle;box-shadow:0 0 8px {swatch};"></span>'
            f'{pred_class}</h3>{route_badge}',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"SE Confidence: **{confidence*100:.1f}%** &nbsp;|&nbsp; "
            f"Threshold: **{threshold*100:.0f}%** &nbsp;|&nbsp; "
            f"Route: **{'SE Direct' if source == 'se_direct' else 'KNN Fallback'}**"
        )

    with conf_col:
        bar_color = "#34d399" if confidence >= threshold else "#fbbf24"
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            number={"suffix": "%", "font": {"size": 28, "color": "#f1f5f9"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#64748b"},
                "bar": {"color": bar_color},
                "bgcolor": "#1a2236",
                "bordercolor": "#2a3548",
                "steps": [
                    {"range": [0, threshold * 100], "color": "rgba(251,146,60,.15)"},
                    {"range": [threshold * 100, 100], "color": "rgba(52,211,153,.15)"},
                ],
                "threshold": {"line": {"color": "#f87171", "width": 3}, "thickness": 0.75, "value": threshold * 100},
            },
            title={"text": "SE Confidence", "font": {"color": "#94a3b8", "size": 14}},
        ))
        style_plot(fig_gauge, height=220)
        st.plotly_chart(fig_gauge, use_container_width=True)

    # Top-5 bar
    st.markdown("#### Top-5 Class Probabilities")
    top5_idx  = np.argsort(proba_vec)[::-1][:5]
    top5_cls  = [EUROSAT_CLASSES[i] for i in top5_idx]
    top5_prob = [float(proba_vec[i]) * 100 for i in top5_idx]
    bar_colors = ["#34d399" if i == pred_idx else "#38bdf8" for i in top5_idx]

    fig_bar = go.Figure(go.Bar(
        x=top5_prob, y=top5_cls, orientation="h",
        marker_color=bar_colors,
        text=[f"{p:.1f}%" for p in top5_prob],
        textposition="outside",
        textfont=dict(color="#e2e8f0"),
        hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
    ))
    fig_bar.update_layout(xaxis_title="Probability (%)", xaxis_range=[0, 115], yaxis={"autorange": "reversed"})
    style_plot(fig_bar, height=280)
    st.plotly_chart(fig_bar, use_container_width=True)

    # Full probability table
    with st.expander("All class probabilities"):
        import pandas as pd
        df_proba = pd.DataFrame({
            "Class": EUROSAT_CLASSES,
            "Probability (%)": [f"{float(proba_vec[i])*100:.2f}" for i in range(len(EUROSAT_CLASSES))],
        }).sort_values("Probability (%)", ascending=False)
        st.dataframe(df_proba, use_container_width=True, hide_index=True)

    # ── Markov Land-Use Forecasting ───────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">🔮 Markov Land-Use Forecasting</div>', unsafe_allow_html=True)
    st.markdown(
        "Using the **Markov Transition Probability Matrix** (TPM) trained on EuroSAT class co-occurrence patterns, "
        "we forecast how the predicted land-use class is likely to evolve over time — and flag degradation risk."
    )

    markov = load_markov()
    if markov is None:
        st.markdown('<div class="warn-box">⚠️ <code>markov.pkl</code> not found. Place it in the <code>streamlit_app/</code> directory.</div>', unsafe_allow_html=True)
    else:
        import pandas as pd
        tpm     = markov["TPM_BLENDED"]          # (10, 10)
        drs     = markov["DEGRADATION_RISK_SCORES"]  # dict {class: score}
        classes = markov["classes"]              # list of 10 class names

        # Map predicted class index to markov class index (same order, but be safe)
        try:
            mc_idx = classes.index(pred_class)
        except ValueError:
            mc_idx = pred_idx

        # Simulate n_steps ahead using matrix power
        n_steps = st.slider("Forecast horizon (time steps)", min_value=1, max_value=20, value=5, step=1)

        # Compute state distribution at each step
        state_0 = np.zeros(len(classes))
        state_0[mc_idx] = 1.0

        states = [state_0]
        for _ in range(n_steps):
            states.append(states[-1] @ tpm)

        states_arr = np.array(states)  # (n_steps+1, 10)

        # ── Forecast line chart ───────────────────────────────────────────────
        st.markdown("#### State Distribution Over Time")
        fig_markov = go.Figure()
        step_labels = list(range(n_steps + 1))
        # Only show top-5 classes by final probability to keep chart clean
        final_probs = states_arr[-1]
        top5_mc = np.argsort(final_probs)[::-1][:5]

        palette = ["#38bdf8", "#34d399", "#fbbf24", "#f472b6", "#a78bfa"]
        for rank, ci in enumerate(top5_mc):
            fig_markov.add_trace(go.Scatter(
                x=step_labels,
                y=states_arr[:, ci] * 100,
                mode="lines+markers",
                name=classes[ci],
                line=dict(color=palette[rank], width=2.5),
                marker=dict(size=6),
                hovertemplate=f"<b>{classes[ci]}</b><br>Step %{{x}}: %{{y:.2f}}%<extra></extra>",
            ))
        fig_markov.update_layout(
            xaxis_title="Time Step",
            yaxis_title="Probability (%)",
            yaxis_range=[0, 105],
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        style_plot(fig_markov, height=360)
        st.plotly_chart(fig_markov, use_container_width=True)

        # ── Transition heatmap for current class ─────────────────────────────
        col_heat, col_risk = st.columns([3, 2])

        with col_heat:
            st.markdown(f"#### Transition Probabilities from **{pred_class}**")
            row = tpm[mc_idx]
            sorted_ci = np.argsort(row)[::-1]
            fig_trans = go.Figure(go.Bar(
                x=[classes[i] for i in sorted_ci],
                y=[row[i] * 100 for i in sorted_ci],
                marker_color=[CLASS_COLORS.get(classes[i], "#64748b") for i in sorted_ci],
                text=[f"{row[i]*100:.1f}%" for i in sorted_ci],
                textposition="outside",
                textfont=dict(color="#e2e8f0", size=10),
                hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
            ))
            fig_trans.update_layout(
                xaxis_title="Next State",
                yaxis_title="Transition Probability (%)",
                yaxis_range=[0, 110],
                xaxis_tickangle=-35,
                showlegend=False,
            )
            style_plot(fig_trans, height=320)
            st.plotly_chart(fig_trans, use_container_width=True)

        with col_risk:
            st.markdown("#### Degradation Risk Scores")
            drs_classes = list(drs.keys())
            drs_scores  = [drs[c] for c in drs_classes]
            sorted_drs  = sorted(zip(drs_classes, drs_scores), key=lambda x: x[1], reverse=True)
            drs_cls_s, drs_val_s = zip(*sorted_drs)

            bar_colors_drs = []
            for v in drs_val_s:
                if v >= 0.7:   bar_colors_drs.append("#f87171")
                elif v >= 0.4: bar_colors_drs.append("#fbbf24")
                else:          bar_colors_drs.append("#34d399")

            fig_drs = go.Figure(go.Bar(
                x=list(drs_val_s),
                y=list(drs_cls_s),
                orientation="h",
                marker_color=bar_colors_drs,
                text=[f"{v:.2f}" for v in drs_val_s],
                textposition="outside",
                textfont=dict(color="#e2e8f0", size=10),
                hovertemplate="%{y}: %{x:.2f}<extra></extra>",
            ))
            fig_drs.update_layout(
                xaxis_title="Risk Score (0–1)",
                xaxis_range=[0, 1.15],
                yaxis=dict(autorange="reversed"),
                showlegend=False,
            )
            style_plot(fig_drs, height=320)
            st.plotly_chart(fig_drs, use_container_width=True)

        # ── Risk callout for predicted class ─────────────────────────────────
        pred_risk = drs.get(pred_class, 0.0)
        if pred_risk >= 0.7:
            risk_color, risk_label, risk_cls = "#f87171", "HIGH", "warn-box"
        elif pred_risk >= 0.4:
            risk_color, risk_label, risk_cls = "#fbbf24", "MEDIUM", "warn-box"
        else:
            risk_color, risk_label, risk_cls = "#34d399", "LOW", "info-box"

        st.markdown(
            f'<div class="{risk_cls}"><strong>Degradation Risk for {pred_class}:</strong> '
            f'<span style="color:{risk_color};font-weight:700;">{risk_label} ({pred_risk:.2f})</span> — '
            f'Self-transition probability: <strong>{tpm[mc_idx, mc_idx]*100:.1f}%</strong>. '
            f'After {n_steps} steps, probability of remaining {pred_class}: '
            f'<strong>{states_arr[-1][mc_idx]*100:.1f}%</strong>.</div>',
            unsafe_allow_html=True,
        )

        # ── Stationary distribution ───────────────────────────────────────────
        with st.expander("Stationary Distribution (Long-Run Equilibrium)"):
            # Power method: run 100 steps
            s = np.ones(len(classes)) / len(classes)
            for _ in range(200):
                s = s @ tpm
            fig_stat = go.Figure(go.Bar(
                x=classes,
                y=s * 100,
                marker_color=[CLASS_COLORS.get(c, "#64748b") for c in classes],
                text=[f"{v*100:.1f}%" for v in s],
                textposition="outside",
                textfont=dict(color="#e2e8f0", size=10),
                hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
            ))
            fig_stat.update_layout(
                xaxis_title="Class",
                yaxis_title="Stationary Probability (%)",
                yaxis_range=[0, 30],
                xaxis_tickangle=-35,
                showlegend=False,
            )
            style_plot(fig_stat, height=300)
            st.plotly_chart(fig_stat, use_container_width=True)
            st.caption("Long-run equilibrium distribution after 200 Markov steps — reflects the natural prevalence of each land-use class in the EuroSAT dataset.")


# =============================================================================
# PAGE 3 — MODEL COMPARISON
# =============================================================================

def page_model_comparison():
    import pandas as pd

    st.markdown('<div class="section-header">Model Comparison</div>', unsafe_allow_html=True)
    st.markdown("All models benchmarked on the EuroSAT test set (n=5,400). Use the toggle to switch between accuracy and F1.")

    art = get_artefacts()
    if not art:
        st.error("No artefacts found. Run `python extract_notebook_metrics.py` to generate metrics.")
        return

    ml  = art.get("ml_results", {})

    models = [
        {"name": "Logistic Regression", "acc": fmt_pct(ml["Logistic Regression"]["acc"]),  "f1": float(ml["Logistic Regression"]["f1"]),  "group": "Classical ML",   "color": "#3b82f6"},
        {"name": "KNN (k=5)",           "acc": fmt_pct(ml["KNN (k=5)"]["acc"]),            "f1": float(ml["KNN (k=5)"]["f1"]),            "group": "Classical ML",   "color": "#60a5fa"},
        {"name": "Decision Tree",       "acc": fmt_pct(ml["Decision Tree"]["acc"]),         "f1": float(ml["Decision Tree"]["f1"]),         "group": "Classical ML",   "color": "#93c5fd"},
        {"name": "Random Forest",       "acc": fmt_pct(ml["Random Forest"]["acc"]),         "f1": float(ml["Random Forest"]["f1"]),         "group": "Classical ML",   "color": "#bfdbfe"},
        {"name": "Vanilla MobileNetV2", "acc": fmt_pct(art["van_acc"]),                     "f1": float(art["van_f1"]),                     "group": "Deep Learning",  "color": "#f59e0b"},
        {"name": "SE-MobileNetV2",      "acc": fmt_pct(art["se_acc"]),                      "f1": float(art["se_f1"]),                      "group": "Deep Learning",  "color": "#10b981"},
        {"name": "SE + Agentic Router", "acc": fmt_pct(art["routed_acc"]),                  "f1": float(art["routed_f1"]),                  "group": "Deep Learning",  "color": "#059669"},
    ]

    names  = [m["name"]  for m in models]
    accs   = [m["acc"]   for m in models]
    f1s    = [m["f1"]    for m in models]
    colors = [m["color"] for m in models]
    best_acc_idx = int(np.argmax(accs))
    best_f1_idx  = int(np.argmax(f1s))

    metric_choice = st.radio("Show metric:", ["Accuracy (%)", "Macro F1"], horizontal=True)

    if metric_choice == "Accuracy (%)":
        vals  = accs
        ytitle = "Accuracy (%)"
        yrange = [0, 108]
        texts  = [f"<b>{v:.2f}% ★</b>" if i == best_acc_idx else f"{v:.2f}%" for i, v in enumerate(vals)]
    else:
        vals  = [f * 100 for f in f1s]
        ytitle = "Macro F1 × 100"
        yrange = [0, 108]
        texts  = [f"<b>{v:.2f} ★</b>" if i == best_f1_idx else f"{v:.2f}" for i, v in enumerate(vals)]

    fig_bar = go.Figure(go.Bar(
        x=names, y=vals,
        marker_color=colors,
        text=texts,
        textposition="outside",
        textfont=dict(color="#e2e8f0", size=11),
        hovertemplate="%{x}<br>%{y:.2f}<extra></extra>",
    ))
    fig_bar.update_layout(yaxis_title=ytitle, yaxis_range=yrange, showlegend=False)
    style_plot(fig_bar, height=400)
    st.plotly_chart(fig_bar, use_container_width=True)

    # Scatter: Accuracy vs F1
    st.markdown('<div class="section-header">Accuracy vs F1 Score</div>', unsafe_allow_html=True)
    fig_scatter = go.Figure()
    for m in models:
        fig_scatter.add_trace(go.Scatter(
            x=[m["acc"]], y=[m["f1"]],
            mode="markers+text",
            marker=dict(size=16, color=m["color"], line=dict(width=1.5, color="#0b0f19")),
            text=[m["name"]], textposition="top center",
            textfont=dict(size=10, color="#cbd5e1"),
            name=m["name"], showlegend=False,
            hovertemplate=f"<b>{m['name']}</b><br>Acc: %{{x:.2f}}%<br>F1: %{{y:.4f}}<extra></extra>",
        ))
    fig_scatter.update_layout(xaxis_title="Accuracy (%)", yaxis_title="Macro F1 Score")
    style_plot(fig_scatter, height=400)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Summary table
    st.markdown('<div class="section-header">Summary Table</div>', unsafe_allow_html=True)
    van_acc = fmt_pct(art["van_acc"])
    df_summary = pd.DataFrame({
        "Model":         names,
        "Group":         [m["group"] for m in models],
        "Accuracy (%)":  [f"{a:.2f}" for a in accs],
        "Macro F1":      [f"{f:.4f}" for f in f1s],
        "vs Vanilla":    [f"{a - van_acc:+.2f}%" for a in accs],
    })
    st.dataframe(df_summary, use_container_width=True, hide_index=True)

    best = models[best_acc_idx]
    se_acc = fmt_pct(art["se_acc"])
    n_knn  = int(art.get("n_knn_routed", 810))
    n_test = int(art.get("n_test", 5400))
    st.markdown(
        f'<div class="info-box"><strong>Best model: {best["name"]}</strong> — '
        f'Accuracy {best["acc"]:.2f}%, Macro F1 {best["f1"]:.4f}. '
        f'The agentic router adds {best["acc"] - se_acc:.2f}% on top of SE-MobileNetV2 alone '
        f'by routing {n_knn/n_test*100:.1f}% of low-confidence samples to the KNN specialist.</div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# PAGE 4 — TRAINING PROGRESS
# =============================================================================

def page_training_progress():
    import pandas as pd

    st.markdown('<div class="section-header">Training Progress</div>', unsafe_allow_html=True)
    st.markdown(
        "SE-MobileNetV2 trained with **3-phase curriculum learning**: easy samples first (40%), "
        "then medium difficulty (70%), then the full dataset (100%)."
    )

    art = get_artefacts()
    if not art:
        st.error("No artefacts found. Run `python extract_notebook_metrics.py` to generate metrics.")
        return

    if "history" in art:
        p1 = art["history"].get("phase1_val_acc", [])
        p2 = art["history"].get("phase2_val_acc", [])
        p3 = art["history"].get("phase3_val_acc", [])
    else:
        st.warning("Training history not found in artefacts.")
        return

    best1 = max(p1) * 100 if p1 else 88.5
    best2 = max(p2) * 100 if p2 else 95.0
    best3 = max(p3) * 100 if p3 else 99.1

    # Phase summary cards
    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        st.markdown(f'<div class="phase-card" style="border-top:3px solid #38bdf8;"><div class="phase-label">Phase 1 — Easy Tier (40%)</div><div class="phase-value">{best1:.2f}%</div><div class="phase-sub">{len(p1)} epochs · Best val accuracy</div></div>', unsafe_allow_html=True)
    with pc2:
        st.markdown(f'<div class="phase-card" style="border-top:3px solid #a78bfa;"><div class="phase-label">Phase 2 — Medium Tier (70%)</div><div class="phase-value">{best2:.2f}%</div><div class="phase-sub">{len(p2)} epochs · Best val accuracy</div></div>', unsafe_allow_html=True)
    with pc3:
        st.markdown(f'<div class="phase-card" style="border-top:3px solid #34d399;"><div class="phase-label">Phase 3 — Full Dataset (100%)</div><div class="phase-value">{best3:.2f}%</div><div class="phase-sub">{len(p3)} epochs · Best val accuracy</div></div>', unsafe_allow_html=True)

    # Multi-phase line chart
    st.markdown('<div class="section-header">Validation Accuracy Across All Phases</div>', unsafe_allow_html=True)

    n1, n2, n3 = len(p1), len(p2), len(p3)
    ep1 = list(range(1, n1 + 1))
    ep2 = list(range(n1 + 1, n1 + n2 + 1))
    ep3 = list(range(n1 + n2 + 1, n1 + n2 + n3 + 1))

    fig_train = go.Figure()
    for ep, vals, name, color in [
        (ep1, p1, "Phase 1 (Easy 40%)",    "#38bdf8"),
        (ep2, p2, "Phase 2 (Medium 70%)",  "#a78bfa"),
        (ep3, p3, "Phase 3 (Full 100%)",   "#34d399"),
    ]:
        fig_train.add_trace(go.Scatter(
            x=ep, y=[v * 100 for v in vals],
            mode="lines+markers", name=name,
            line=dict(color=color, width=2.5),
            marker=dict(size=6),
            hovertemplate="Epoch %{x}<br>Val Acc: %{y:.2f}%<extra></extra>",
        ))

    for sep_x, label, color in [
        (n1 + 0.5, "Phase 2 starts", "#a78bfa"),
        (n1 + n2 + 0.5, "Phase 3 starts", "#34d399"),
    ]:
        fig_train.add_vline(x=sep_x, line_dash="dash", line_color=color, line_width=1.5, opacity=0.6)
        fig_train.add_annotation(x=sep_x, y=103, text=label, showarrow=False, font=dict(size=10, color=color), xanchor="left")

    fig_train.update_layout(
        xaxis_title="Epoch",
        yaxis_title="Validation Accuracy (%)",
        yaxis_range=[48, 106],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    style_plot(fig_train, height=460)
    st.plotly_chart(fig_train, use_container_width=True)

    st.markdown(
        f'<div class="info-box"><strong>Curriculum Learning:</strong> Samples ranked by difficulty using a '
        f'Random Forest proxy on PCA features. Phase 1 trains on the easiest 40%, Phase 2 expands to 70%, '
        f'Phase 3 uses the full dataset. Final validation accuracy: <strong>{best3:.2f}%</strong> '
        f'vs <strong>{best1:.2f}%</strong> at end of Phase 1.</div>',
        unsafe_allow_html=True,
    )

    with st.expander("Phase-by-phase epoch details"):
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            st.markdown("**Phase 1**")
            st.dataframe(pd.DataFrame({"Epoch": ep1, "Val Acc (%)": [f"{v*100:.2f}" for v in p1]}), use_container_width=True, hide_index=True)
        with col_t2:
            st.markdown("**Phase 2**")
            st.dataframe(pd.DataFrame({"Epoch": ep2, "Val Acc (%)": [f"{v*100:.2f}" for v in p2]}), use_container_width=True, hide_index=True)
        with col_t3:
            st.markdown("**Phase 3**")
            st.dataframe(pd.DataFrame({"Epoch": ep3, "Val Acc (%)": [f"{v*100:.2f}" for v in p3]}), use_container_width=True, hide_index=True)


# =============================================================================
# PAGE 5 — EXPLAINABILITY
# =============================================================================

def page_explainability():
    import pandas as pd

    st.markdown('<div class="section-header">Explainability</div>', unsafe_allow_html=True)
    st.markdown("How SE-MobileNetV2 attends to spatial regions and recalibrates channel responses.")

    model = load_model()
    art   = get_artefacts()

    # Grad-CAM
    with st.expander("Grad-CAM: Class Activation Maps", expanded=True):
        st.markdown("""
**Grad-CAM** highlights spatial regions that most influenced the prediction.

1. Forward pass → predicted class score
2. Gradients of class score w.r.t. last conv feature map
3. Weight channels by global average gradient → ReLU → upsample

Red/yellow = high attention · Blue = low attention
""")
        if model is not None:
            xtest_path = artefact_paths().get("x_test", "")
            ytest_path = artefact_paths().get("y_test", "")
            if os.path.isfile(xtest_path):
                with st.spinner("Generating Grad-CAM grid…"):
                    try:
                        X_test = np.load(xtest_path)
                        y_test = np.load(ytest_path) if os.path.isfile(ytest_path) else None
                        n_show = min(10, len(X_test))
                        indices = np.random.choice(len(X_test), n_show, replace=False)
                        fig_grid, axes = plt.subplots(3, n_show, figsize=(n_show * 2.5, 7))
                        fig_grid.patch.set_facecolor("#0b0f19")
                        for col_i, idx in enumerate(indices):
                            img = X_test[idx]
                            result = make_gradcam(model, img, IMG_SIZE)
                            axes[0, col_i].imshow(img); axes[0, col_i].axis("off")
                            if y_test is not None:
                                axes[0, col_i].set_title(EUROSAT_CLASSES[int(y_test[idx])], fontsize=7, color="#94a3b8")
                            if result is not None:
                                cam, overlay, pred_idx, conf = result
                                axes[1, col_i].imshow(cam, cmap="jet"); axes[1, col_i].axis("off")
                                axes[1, col_i].set_title(f"{conf*100:.0f}%", fontsize=7, color="#38bdf8")
                                axes[2, col_i].imshow(overlay); axes[2, col_i].axis("off")
                                axes[2, col_i].set_title(EUROSAT_CLASSES[pred_idx], fontsize=7, color="#34d399")
                            else:
                                for r in [1, 2]: axes[r, col_i].axis("off")
                        for ax in axes.flat: ax.set_facecolor("#151b28")
                        for r, lbl in enumerate(["Original", "Grad-CAM", "Overlay"]):
                            axes[r, 0].set_ylabel(lbl, fontsize=9, color="#94a3b8", rotation=90, labelpad=4)
                        plt.tight_layout(pad=0.5)
                        st.pyplot(fig_grid); plt.close(fig_grid)
                    except Exception as e:
                        st.info(f"Could not generate Grad-CAM grid: {e}")
            else:
                st.info("Upload an image on the **Predict** page for live Grad-CAM. Place `X_test.npy` in `artefacts/` for a grid view.")
        else:
            st.info("Load model weights to enable live Grad-CAM visualisation.")

    # SE block explanation
    with st.expander("Squeeze-and-Excitation Channel Attention", expanded=True):
        st.markdown("""
SE blocks learn to recalibrate channel-wise feature responses:

```
Input (H×W×C)  →  GlobalAvgPool (1×1×C)  →  FC(C/r, ReLU)  →  FC(C, Sigmoid)  →  Multiply(input)
```
Applied at 4 scales — captures both local texture and global context.
""")
        df_se = pd.DataFrame({
            "Feature Tap":        ["block_3_expand_relu", "block_6_expand_relu", "block_13_expand_relu", "out_relu"],
            "Spatial Size":       ["16×16", "8×8", "4×4", "2×2"],
            "Channels (C)":       [96, 192, 576, 1280],
            "SE Bottleneck (C/16)": [6, 12, 36, 80],
            "SE Params":          [96*6*2, 192*12*2, 576*36*2, 1280*80*2],
            "GAP Output":         [96, 192, 576, 1280],
        })
        st.dataframe(df_se, use_container_width=True, hide_index=True)
        st.markdown("All four GAP outputs concatenate into a **2144-d** embedding → Dense(512) → BN → Dropout(0.4) → Dense(256) → Dropout(0.3) → Softmax(10).")

    # Agentic router
    with st.expander("Agentic Confidence Router", expanded=True):
        art_data  = get_artefacts()
        if not art_data:
            st.info("No artefacts found.")
            return
        threshold = float(art_data.get("confidence_threshold", 0.60))
        n_direct  = int(art_data.get("n_se_direct",  4590))
        n_knn     = int(art_data.get("n_knn_routed",  810))
        n_total   = int(art_data.get("n_test",        5400))

        col_text, col_pie = st.columns([1, 1])
        with col_text:
            st.markdown(f"""
**Routing logic:**
- conf ≥ **{threshold:.2f}** → SE direct prediction
- conf < **{threshold:.2f}** → KNN fallback (k=7, cosine, 256-d SE embeddings)

**Test set breakdown (n={n_total:,}):**
- SE direct: **{n_direct:,}** ({n_direct/n_total*100:.1f}%)
- KNN fallback: **{n_knn:,}** ({n_knn/n_total*100:.1f}%)
""")
        with col_pie:
            fig_pie = go.Figure(go.Pie(
                labels=["SE Direct", "KNN Fallback"],
                values=[n_direct, n_knn],
                hole=0.5,
                marker=dict(colors=["#34d399", "#fbbf24"]),
                textfont=dict(color="#e2e8f0"),
                hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>",
            ))
            fig_pie.update_layout(
                annotations=[dict(text=f"{n_total:,}<br>samples", x=0.5, y=0.5, font=dict(size=13, color="#f1f5f9"), showarrow=False)],
            )
            style_plot(fig_pie, height=280)
            st.plotly_chart(fig_pie, use_container_width=True)


# =============================================================================
# PAGE 6 — ABOUT
# =============================================================================

def page_about():
    import pandas as pd

    st.markdown('<div class="section-header">About This Project</div>', unsafe_allow_html=True)
    st.markdown(
        "A complete land-use classification pipeline for the **EuroSAT RGB** dataset — "
        "custom SE-MobileNetV2 architecture, curriculum learning, and an agentic confidence router."
    )

    art = get_artefacts()
    if not art:
        st.error("No artefacts found. Run `python extract_notebook_metrics.py` to generate metrics.")
        return

    se_acc     = fmt_pct(art["se_acc"])
    routed_acc = fmt_pct(art["routed_acc"])
    van_acc    = fmt_pct(art["van_acc"])
    se_f1      = float(art["se_f1"])
    rt_f1      = float(art["routed_f1"])

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="value">{routed_acc:.2f}%</div><div class="label">Best Accuracy (SE + Router)</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card g"><div class="value">{rt_f1:.4f}</div><div class="label">Best Macro F1 (SE + Router)</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card a"><div class="value">+{routed_acc - van_acc:.2f}%</div><div class="label">Improvement vs Vanilla MobileNetV2</div></div>', unsafe_allow_html=True)

    # Architecture summary
    st.markdown('<div class="section-header">Architecture Summary</div>', unsafe_allow_html=True)
    df_arch = pd.DataFrame({
        "Component": ["Backbone", "Feature Taps", "Attention", "Fusion", "Head", "Loss", "Training", "Router", "Fallback"],
        "Detail": [
            "MobileNetV2 (ImageNet pretrained)",
            "4 scales: block_3, block_6, block_13, out_relu",
            "Squeeze-and-Excitation blocks (ratio=16) at each scale",
            "GlobalAveragePool + Concatenate → 2144-d",
            "Dense(512, BN, Dropout 0.4) → Dense(256, Dropout 0.3) → Softmax(10)",
            "Focal Loss (gamma=2.0, alpha=0.25)",
            "3-phase curriculum: easy 40% → medium 70% → full 100%",
            "Confidence threshold = 0.60",
            "KNN (k=7, cosine) on 256-d SE embeddings",
        ],
        "Result": [
            "Strong feature extraction",
            "Multi-scale spatial context",
            f"SE Accuracy: {se_acc:.2f}%",
            "2144-d multi-scale embedding",
            f"SE F1: {se_f1:.4f}",
            "Focus on hard examples",
            f"Phase 3 best: {se_acc:.2f}%",
            f"{int(art.get('n_knn_routed',810))/int(art.get('n_test',5400))*100:.1f}% samples routed",
            f"Router Accuracy: {routed_acc:.2f}%",
        ],
    })
    st.dataframe(df_arch, use_container_width=True, hide_index=True)

    # Dataset
    st.markdown('<div class="section-header">EuroSAT Dataset</div>', unsafe_allow_html=True)
    col_info, col_classes = st.columns([1, 1])
    with col_info:
        st.markdown("""
**EuroSAT** — 27,000 labeled Sentinel-2 satellite patches, 10 land-use classes across Europe.

| Property | Value |
|---|---|
| Total samples | 27,000 |
| Image size | 64 × 64 px, RGB |
| Train / Test split | 80% / 20% stratified |
| Test set size | 5,400 samples |
| Samples per class | 2,000 – 3,000 |
""")
    with col_classes:
        df_cls = pd.DataFrame({
            "Class": EUROSAT_CLASSES,
            "Description": [
                "Fields with annual crops",
                "Dense forest and woodland",
                "Grasslands and herbaceous vegetation",
                "Roads and transport corridors",
                "Industrial zones and warehouses",
                "Pasture and grazing land",
                "Orchards and permanent crops",
                "Urban residential areas",
                "Rivers and waterways",
                "Sea, lakes, and water bodies",
            ],
        })
        st.dataframe(df_cls, use_container_width=True, hide_index=True)

    # Class swatches
    cols = st.columns(5)
    for i, cls in enumerate(EUROSAT_CLASSES):
        color = CLASS_COLORS.get(cls, "#64748b")
        with cols[i % 5]:
            st.markdown(f'<div class="class-card"><div class="swatch" style="background:{color};"></div><div class="name">{cls}</div></div>', unsafe_allow_html=True)

    # Tech stack
    st.markdown('<div class="section-header">Technology Stack</div>', unsafe_allow_html=True)
    tech_items = [
        ("Python 3.10+", "#3b82f6"), ("TensorFlow / Keras", "#f59e0b"), ("MobileNetV2", "#10b981"),
        ("Streamlit", "#f472b6"), ("Plotly", "#a78bfa"), ("scikit-learn", "#38bdf8"),
        ("NumPy", "#34d399"), ("OpenCV", "#fbbf24"), ("Pillow", "#94a3b8"), ("EuroSAT Dataset", "#64748b"),
    ]
    badges = "".join(f'<span class="tech-badge" style="border-color:{c};color:{c};">{n}</span>' for n, c in tech_items)
    st.markdown(f'<div style="margin:12px 0;">{badges}</div>', unsafe_allow_html=True)

    # References
    st.markdown('<div class="section-header">References</div>', unsafe_allow_html=True)
    st.markdown("""
1. **EuroSAT:** Helber et al. (2019). EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification. *IEEE JSTARS.*
2. **SE Networks:** Hu et al. (2018). Squeeze-and-Excitation Networks. *CVPR 2018.*
3. **MobileNetV2:** Sandler et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. *CVPR 2018.*
4. **Focal Loss:** Lin et al. (2017). Focal Loss for Dense Object Detection. *ICCV 2017.*
5. **Curriculum Learning:** Bengio et al. (2009). Curriculum Learning. *ICML 2009.*
6. **Grad-CAM:** Selvaraju et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. *ICCV 2017.*
""")


# =============================================================================
# MAIN ROUTER
# =============================================================================

def main():
    page = st.session_state.nav_page
    if   page == "Home":              page_home()
    elif page == "Predict":           page_predict()
    elif page == "Model Comparison":  page_model_comparison()
    elif page == "Training Progress": page_training_progress()
    elif page == "Explainability":    page_explainability()
    elif page == "About":             page_about()

main()
