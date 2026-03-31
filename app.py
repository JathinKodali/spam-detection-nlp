"""
Spam Message Detection — Streamlit Application
================================================
A premium, review-ready dashboard for SMS spam classification.

Run:  streamlit run app.py
"""

import pandas as pd
import streamlit as st

from model import (
    load_or_download_data,
    predict_batch,
    predict_message,
    preprocess_text,
    train_model,
)
from utils import (
    class_distribution_chart,
    confidence_gauge,
    confusion_matrix_chart,
    generate_wordcloud,
    message_length_distribution,
    roc_curve_chart,
    top_features_chart,
)

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="SMS Spam Detector · NLP Project",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Inject premium CSS
# ──────────────────────────────────────────────
st.markdown(
    """
<style>
/* ============ FONTS ============ */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif !important;
}

/* ============ ANIMATED BACKGROUND ============ */
.stApp {
    background: #06080f;
    overflow-x: hidden;
}
.stApp::before {
    content: '';
    position: fixed;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background:
        radial-gradient(ellipse 600px 600px at 20% 30%, rgba(99, 102, 241, 0.08) 0%, transparent 70%),
        radial-gradient(ellipse 500px 500px at 75% 60%, rgba(236, 72, 153, 0.07) 0%, transparent 70%),
        radial-gradient(ellipse 400px 400px at 50% 80%, rgba(6, 182, 212, 0.06) 0%, transparent 70%);
    z-index: 0;
    animation: bgPulse 12s ease-in-out infinite alternate;
    pointer-events: none;
}
@keyframes bgPulse {
    0%   { transform: translate(0, 0) scale(1); }
    50%  { transform: translate(20px, -30px) scale(1.05); }
    100% { transform: translate(-10px, 15px) scale(0.98); }
}

/* ============ SIDEBAR ============ */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c1018 0%, #0a0e1a 100%) !important;
    border-right: 1px solid rgba(99, 102, 241, 0.12);
}
section[data-testid="stSidebar"] .stRadio > label {
    font-weight: 600;
    letter-spacing: 0.5px;
    color: #a0aec0 !important;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    padding: 10px 16px !important;
    border-radius: 12px !important;
    margin-bottom: 4px !important;
    transition: all 0.3s ease !important;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background: rgba(99, 102, 241, 0.08) !important;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-checked="true"] {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(236, 72, 153, 0.1)) !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
}

/* ============ GLASSMORPHIC CARDS ============ */
.glass-card {
    background: linear-gradient(145deg,
        rgba(255, 255, 255, 0.04) 0%,
        rgba(255, 255, 255, 0.01) 100%);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 20px;
    padding: 28px;
    text-align: center;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    position: relative;
    overflow: hidden;
}
.glass-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.03), transparent);
    transition: left 0.6s ease;
}
.glass-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 20px 60px rgba(99, 102, 241, 0.15);
    border-color: rgba(99, 102, 241, 0.25);
}
.glass-card:hover::before {
    left: 100%;
}

/* Metric card variants */
.metric-value {
    font-size: 2.5rem;
    font-weight: 900;
    letter-spacing: -1px;
    line-height: 1;
    margin-bottom: 8px;
}
.metric-value.gradient-blue {
    background: linear-gradient(135deg, #6366f1, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-value.gradient-green {
    background: linear-gradient(135deg, #10b981, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-value.gradient-red {
    background: linear-gradient(135deg, #f43f5e, #fb7185);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-value.gradient-gold {
    background: linear-gradient(135deg, #f59e0b, #fbbf24);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-label {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 600;
}
.metric-icon {
    font-size: 1.5rem;
    margin-bottom: 6px;
    display: block;
}

/* ============ HERO ============ */
.hero-section {
    text-align: center;
    padding: 50px 20px 30px;
    position: relative;
}
.hero-section h1 {
    font-size: 3.8rem;
    font-weight: 900;
    letter-spacing: -2px;
    line-height: 1.1;
    margin-bottom: 12px;
    background: linear-gradient(135deg, #6366f1, #ec4899, #06b6d4, #f59e0b);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientShift 6s ease infinite;
}
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.hero-subtitle {
    color: #94a3b8;
    font-size: 1.15rem;
    max-width: 680px;
    margin: 0 auto;
    line-height: 1.7;
}
.hero-divider {
    width: 60px;
    height: 3px;
    background: linear-gradient(90deg, #6366f1, #ec4899);
    border-radius: 10px;
    margin: 20px auto 0;
}

/* ============ PIPELINE BADGES ============ */
.pipeline-row {
    display: flex;
    gap: 14px;
    flex-wrap: wrap;
    justify-content: center;
    margin: 30px 0;
}
.pipeline-step {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.07);
    padding: 10px 22px;
    border-radius: 50px;
    font-size: 0.82rem;
    color: #cbd5e1;
    font-weight: 500;
    letter-spacing: 0.3px;
    display: flex;
    align-items: center;
    gap: 6px;
    transition: all 0.3s ease;
}
.pipeline-step:hover {
    border-color: rgba(99, 102, 241, 0.4);
    background: rgba(99, 102, 241, 0.08);
    transform: translateY(-2px);
}
.pipeline-arrow {
    color: #4b5563;
    font-size: 0.8rem;
}

/* ============ RESULT BADGES ============ */
.result-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 30px;
    animation: fadeUp 0.6s cubic-bezier(0.16, 1, 0.3, 1);
}
@keyframes fadeUp {
    0%   { opacity: 0; transform: translateY(30px) scale(0.95); }
    100% { opacity: 1; transform: translateY(0) scale(1); }
}
.result-badge-large {
    display: inline-flex;
    align-items: center;
    gap: 14px;
    padding: 20px 44px;
    border-radius: 60px;
    font-size: 1.6rem;
    font-weight: 800;
    letter-spacing: 1.5px;
}
.badge-spam {
    background: linear-gradient(135deg, rgba(244, 63, 94, 0.12), rgba(244, 63, 94, 0.04));
    border: 2px solid rgba(244, 63, 94, 0.5);
    color: #fb7185;
    box-shadow: 0 0 40px rgba(244, 63, 94, 0.15), inset 0 0 40px rgba(244, 63, 94, 0.05);
}
.badge-ham {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.12), rgba(16, 185, 129, 0.04));
    border: 2px solid rgba(16, 185, 129, 0.5);
    color: #34d399;
    box-shadow: 0 0 40px rgba(16, 185, 129, 0.15), inset 0 0 40px rgba(16, 185, 129, 0.05);
}
.result-detail {
    color: #64748b;
    font-size: 0.85rem;
    margin-top: 16px;
    font-family: 'JetBrains Mono', monospace;
    background: rgba(255,255,255,0.02);
    padding: 10px 20px;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.05);
    max-width: 500px;
    word-break: break-word;
}

/* ============ SECTION HEADERS ============ */
.section-title {
    font-size: 1.8rem;
    font-weight: 800;
    color: #f1f5f9;
    letter-spacing: -0.5px;
    margin-bottom: 4px;
}
.section-desc {
    color: #64748b;
    font-size: 0.95rem;
    margin-bottom: 28px;
    line-height: 1.6;
}

/* ============ ABOUT BOX ============ */
.about-box {
    background: linear-gradient(145deg, rgba(99, 102, 241, 0.06), rgba(236, 72, 153, 0.03));
    border: 1px solid rgba(99, 102, 241, 0.12);
    border-radius: 20px;
    padding: 32px;
    margin-top: 16px;
}
.about-box h3 {
    color: #e2e8f0;
    font-size: 1.2rem;
    margin-bottom: 16px;
}
.about-step {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    margin-bottom: 14px;
}
.about-num {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    width: 30px;
    height: 30px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.8rem;
    flex-shrink: 0;
}
.about-text {
    color: #94a3b8;
    font-size: 0.92rem;
    line-height: 1.5;
}
.about-text strong {
    color: #c4b5fd;
}

/* ============ EXAMPLE CARDS ============ */
.example-card {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 12px;
    transition: all 0.3s ease;
}
.example-card:hover {
    border-color: rgba(99, 102, 241, 0.3);
    background: rgba(99, 102, 241, 0.04);
}
.example-tag {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 4px 12px;
    border-radius: 20px;
    flex-shrink: 0;
}
.tag-spam { background: rgba(244,63,94,0.15); color: #fb7185; }
.tag-ham  { background: rgba(16,185,129,0.15); color: #34d399; }
.example-text {
    color: #94a3b8;
    font-size: 0.88rem;
    font-family: 'JetBrains Mono', monospace;
}

/* ============ INPUTS ============ */
.stTextArea textarea {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(99, 102, 241, 0.15) !important;
    border-radius: 16px !important;
    color: #e2e8f0 !important;
    font-size: 1rem !important;
    font-family: 'Inter', sans-serif !important;
    padding: 16px !important;
    transition: all 0.3s ease !important;
}
.stTextArea textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
}
.stTextArea textarea::placeholder {
    color: #475569 !important;
}

/* Primary button */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    border-radius: 14px !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    padding: 12px 28px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4) !important;
}

/* File uploader */
div[data-testid="stFileUploader"] {
    background: rgba(99, 102, 241, 0.03);
    border: 2px dashed rgba(99, 102, 241, 0.15);
    border-radius: 20px;
    padding: 20px;
    transition: all 0.3s ease;
}
div[data-testid="stFileUploader"]:hover {
    border-color: rgba(99, 102, 241, 0.3);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(255,255,255,0.02);
    border-radius: 14px;
    padding: 6px;
    border: 1px solid rgba(255,255,255,0.05);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(139, 92, 246, 0.1)) !important;
}

/* Dataframe */
.stDataFrame {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.06) !important;
}

/* Metric built-in */
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 16px !important;
}

/* Info / warning boxes */
.stAlert {
    border-radius: 16px !important;
}

/* Divider */
hr {
    border-color: rgba(255,255,255,0.05) !important;
    margin: 32px 0 !important;
}

/* Hide streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {
    background: rgba(6, 8, 15, 0.8) !important;
    backdrop-filter: blur(10px) !important;
}

/* ============ SIDEBAR LOGO AREA ============ */
.sidebar-logo {
    text-align: center;
    padding: 10px 0 20px;
}
.sidebar-logo-icon {
    font-size: 2.6rem;
    display: block;
    margin-bottom: 8px;
}
.sidebar-logo-title {
    font-size: 1.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6366f1, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}
.sidebar-logo-sub {
    font-size: 0.68rem;
    color: #475569;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 2px;
}
.sidebar-footer {
    text-align: center;
    color: #334155;
    font-size: 0.72rem;
    padding-top: 12px;
    line-height: 1.7;
}

/* ============ WORD CLOUD CONTAINER ============ */
.wc-container {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 20px;
    padding: 20px;
    text-align: center;
}
.wc-label {
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.wc-label-spam { color: #fb7185; }
.wc-label-ham  { color: #34d399; }

/* ============ CHART TITLE BAR ============ */
.chart-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 16px;
}
.chart-header-icon {
    font-size: 1.3rem;
}
.chart-header-text {
    font-size: 1.1rem;
    font-weight: 700;
    color: #e2e8f0;
}
</style>
""",
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────
# Load data & train model (cached)
# ──────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset …")
def get_data():
    return load_or_download_data()


@st.cache_resource(show_spinner="Training model …")
def get_model_and_metrics(_df):
    return train_model(_df)


df = get_data()
model, vectorizer, metrics = get_model_and_metrics(df)


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-logo">
            <span class="sidebar-logo-icon">🛡️</span>
            <div class="sidebar-logo-title">Spam Detector</div>
            <div class="sidebar-logo-sub">NLP Project</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠  Overview",
            "🔍  Live Prediction",
            "📊  Dataset Explorer",
            "📈  Model Performance",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        """
        <div class="sidebar-footer">
            TF-IDF + Logistic Regression<br>
            SMS Spam Collection Dataset<br><br>
            Built with ❤️ using Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE: Overview
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if page == "🏠  Overview":

    # Hero
    st.markdown(
        """
        <div class="hero-section">
            <h1>SMS Spam Detector</h1>
            <p class="hero-subtitle">
                An intelligent NLP system that classifies SMS messages as spam or legitimate
                using TF-IDF feature extraction and Logistic Regression — achieving <strong>96%+ accuracy</strong>.
            </p>
            <div class="hero-divider"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Pipeline badges
    st.markdown(
        """
        <div class="pipeline-row">
            <div class="pipeline-step">📥 Data Collection</div>
            <span class="pipeline-arrow">→</span>
            <div class="pipeline-step">🧹 Preprocessing</div>
            <span class="pipeline-arrow">→</span>
            <div class="pipeline-step">📐 TF-IDF Vectorizer</div>
            <span class="pipeline-arrow">→</span>
            <div class="pipeline-step">🤖 Logistic Regression</div>
            <span class="pipeline-arrow">→</span>
            <div class="pipeline-step">✅ Classification</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")

    # Metric cards
    total = len(df)
    spam_count = int((df["label"] == "spam").sum())
    ham_count = int((df["label"] == "ham").sum())
    acc = metrics["accuracy"]

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "📬", f"{total:,}", "Total Messages", "gradient-blue"),
        (c2, "✅", f"{ham_count:,}", "Ham Messages", "gradient-green"),
        (c3, "🚫", f"{spam_count:,}", "Spam Messages", "gradient-red"),
        (c4, "🎯", f"{acc*100:.2f}%", "Model Accuracy", "gradient-gold"),
    ]
    for col, icon, val, lbl, grad in cards:
        col.markdown(
            f"""
            <div class="glass-card">
                <span class="metric-icon">{icon}</span>
                <div class="metric-value {grad}">{val}</div>
                <div class="metric-label">{lbl}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")

    # Distribution chart
    st.plotly_chart(class_distribution_chart(df), use_container_width=True)

    st.markdown("---")

    # About
    st.markdown(
        """
        <div class="about-box">
            <h3>📄 How It Works</h3>
            <div class="about-step">
                <div class="about-num">1</div>
                <div class="about-text"><strong>Data Collection</strong> — Uses the SMS Spam Collection dataset with 5,572 real-world messages.</div>
            </div>
            <div class="about-step">
                <div class="about-num">2</div>
                <div class="about-text"><strong>Text Preprocessing</strong> — Lowercases text, strips URLs, numbers, punctuation, and English stopwords.</div>
            </div>
            <div class="about-step">
                <div class="about-num">3</div>
                <div class="about-text"><strong>Feature Extraction</strong> — Transforms cleaned text to 3,000-dimensional TF-IDF vectors.</div>
            </div>
            <div class="about-step">
                <div class="about-num">4</div>
                <div class="about-text"><strong>Classification</strong> — Logistic Regression trained on TF-IDF features for binary classification.</div>
            </div>
            <div class="about-step">
                <div class="about-num">5</div>
                <div class="about-text"><strong>Evaluation</strong> — Measured via accuracy, precision, recall, F1-score, and ROC-AUC.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE: Live Prediction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "🔍  Live Prediction":
    st.markdown(
        '<div class="section-title">🔍 Live Spam Detection</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-desc">Type or paste a message below to classify it instantly. You can also upload a CSV for batch predictions.</div>',
        unsafe_allow_html=True,
    )

    tab_single, tab_batch = st.tabs(["✉️  Single Message", "📂  Batch (CSV Upload)"])

    # ── Single ──
    with tab_single:
        user_msg = st.text_area(
            "Enter your message",
            height=140,
            placeholder="e.g. Congratulations! You have won a free vacation. Call now to claim!",
        )

        col_btn, _ = st.columns([1, 3])
        with col_btn:
            classify_btn = st.button(
                "🚀  Classify Message", use_container_width=True, type="primary"
            )

        if classify_btn and user_msg.strip():
            label, conf = predict_message(user_msg, model, vectorizer)
            badge_cls = "badge-spam" if label == "Spam" else "badge-ham"
            icon = "🚫" if label == "Spam" else "✅"

            c_left, c_right = st.columns([1, 1])
            with c_left:
                st.markdown(
                    f"""
                    <div class="result-container">
                        <div class="result-badge-large {badge_cls}">{icon}  {label.upper()}</div>
                        <div class="result-detail">Processed: {preprocess_text(user_msg)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c_right:
                st.plotly_chart(
                    confidence_gauge(conf, label), use_container_width=True
                )

        elif classify_btn:
            st.warning("⚠️ Please enter a message first.")

        st.markdown("---")
        st.markdown(
            '<div class="chart-header"><span class="chart-header-icon">💡</span><span class="chart-header-text">Try These Examples</span></div>',
            unsafe_allow_html=True,
        )
        examples = [
            ("SPAM", "tag-spam", "WINNER!! As a valued customer you have been selected to receive a £900 prize reward!"),
            ("HAM", "tag-ham", "Hey, are you coming to class today?"),
            ("SPAM", "tag-spam", "Free entry in 2 a wkly comp to win FA Cup final tkts. Text FA to 87121."),
            ("HAM", "tag-ham", "I'll be there in 10 mins. Save me a seat."),
            ("SPAM", "tag-spam", "Urgent! You have won a 1 week free membership. Call now to claim!"),
            ("HAM", "tag-ham", "Can you pick up some groceries on the way home?"),
        ]
        for tag, tag_cls, ex in examples:
            st.markdown(
                f"""
                <div class="example-card">
                    <span class="example-tag {tag_cls}">{tag}</span>
                    <span class="example-text">{ex}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Batch ──
    with tab_batch:
        st.markdown(
            "Upload a CSV with a **`message`** column to classify all rows at once."
        )
        uploaded = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded is not None:
            try:
                batch_df = pd.read_csv(uploaded)
                # Auto-detect message column
                if "message" not in batch_df.columns:
                    for candidate in ["text", "sms", "msg", "v2", "content", "Message", "Text"]:
                        if candidate in batch_df.columns:
                            batch_df = batch_df.rename(columns={candidate: "message"})
                            break
                if "message" not in batch_df.columns:
                    st.error("❌ CSV must contain a **message** column.")
                else:
                    labels, confs = predict_batch(
                        batch_df["message"].astype(str), model, vectorizer
                    )
                    batch_df["Prediction"] = labels
                    batch_df["Confidence"] = [f"{c*100:.1f}%" for c in confs]

                    spam_n = labels.count("Spam")
                    ham_n = labels.count("Ham")

                    c1, c2, c3 = st.columns(3)
                    c1.metric("📊 Total", len(labels))
                    c2.metric("🚫 Spam Detected", spam_n)
                    c3.metric("✅ Ham Messages", ham_n)

                    st.markdown("")
                    st.dataframe(batch_df, use_container_width=True, height=400)

                    csv_bytes = batch_df.to_csv(index=False).encode()
                    st.download_button(
                        "⬇️  Download Results",
                        csv_bytes,
                        file_name="spam_predictions.csv",
                        mime="text/csv",
                    )
            except Exception as e:
                st.error(f"Error: {e}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE: Dataset Explorer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "📊  Dataset Explorer":
    st.markdown(
        '<div class="section-title">📊 Dataset Explorer</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-desc">Interactive visualizations and analysis of the SMS Spam Collection dataset.</div>',
        unsafe_allow_html=True,
    )

    # Quick stats
    c1, c2, c3, c4 = st.columns(4)
    avg_len = df["message"].str.len().mean()
    spam_pct = (df["label"] == "spam").mean() * 100
    spam_avg = df[df["label"] == "spam"]["message"].str.len().mean()
    ham_avg = df[df["label"] == "ham"]["message"].str.len().mean()

    c1.metric("📊 Total Samples", f"{len(df):,}")
    c2.metric("📏 Avg Length", f"{avg_len:.0f} chars")
    c3.metric("🚫 Spam %", f"{spam_pct:.1f}%")
    c4.metric("📐 Spam Avg Len", f"{spam_avg:.0f} chars")

    st.markdown("")

    # Message length distribution
    st.markdown(
        '<div class="chart-header"><span class="chart-header-icon">📏</span><span class="chart-header-text">Message Length Distribution</span></div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(message_length_distribution(df), use_container_width=True)

    st.markdown("---")

    # Word clouds
    st.markdown(
        '<div class="chart-header"><span class="chart-header-icon">☁️</span><span class="chart-header-text">Word Clouds</span></div>',
        unsafe_allow_html=True,
    )
    wc1, wc2 = st.columns(2)
    with wc1:
        st.markdown(
            '<div class="wc-container"><div class="wc-label wc-label-spam">🚫 Spam Messages</div>',
            unsafe_allow_html=True,
        )
        spam_texts = df[df["label"] == "spam"]["message"].apply(preprocess_text)
        st.image(generate_wordcloud(spam_texts, "Spam"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with wc2:
        st.markdown(
            '<div class="wc-container"><div class="wc-label wc-label-ham">✅ Ham Messages</div>',
            unsafe_allow_html=True,
        )
        ham_texts = df[df["label"] == "ham"]["message"].apply(preprocess_text)
        st.image(generate_wordcloud(ham_texts, "Ham"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # Top TF-IDF features
    st.markdown(
        '<div class="chart-header"><span class="chart-header-icon">🏷️</span><span class="chart-header-text">Top TF-IDF Feature Coefficients</span></div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        top_features_chart(metrics["top_spam_words"], metrics["top_ham_words"]),
        use_container_width=True,
    )

    st.markdown("---")

    # Sample data
    st.markdown(
        '<div class="chart-header"><span class="chart-header-icon">📋</span><span class="chart-header-text">Sample Data</span></div>',
        unsafe_allow_html=True,
    )
    label_filter = st.selectbox("Filter by label", ["All", "ham", "spam"])
    display_df = df if label_filter == "All" else df[df["label"] == label_filter]
    st.dataframe(display_df.head(50), use_container_width=True, height=350)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE: Model Performance
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "📈  Model Performance":
    st.markdown(
        '<div class="section-title">📈 Model Performance</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-desc">Comprehensive evaluation of the Logistic Regression classifier on the test set.</div>',
        unsafe_allow_html=True,
    )

    report = metrics["report"]

    # Metric cards
    c1, c2, c3, c4 = st.columns(4)
    perf_cards = [
        (c1, "🎯", f"{metrics['accuracy']*100:.2f}%", "Accuracy", "gradient-gold"),
        (c2, "🔬", f"{report['Spam']['precision']*100:.2f}%", "Precision (Spam)", "gradient-blue"),
        (c3, "📡", f"{report['Spam']['recall']*100:.2f}%", "Recall (Spam)", "gradient-red"),
        (c4, "⚡", f"{report['Spam']['f1-score']*100:.2f}%", "F1-Score (Spam)", "gradient-green"),
    ]
    for col, icon, val, lbl, grad in perf_cards:
        col.markdown(
            f"""
            <div class="glass-card">
                <span class="metric-icon">{icon}</span>
                <div class="metric-value {grad}">{val}</div>
                <div class="metric-label">{lbl}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")

    # Charts
    left, right = st.columns(2)
    with left:
        st.markdown(
            '<div class="chart-header"><span class="chart-header-icon">🔢</span><span class="chart-header-text">Confusion Matrix</span></div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            confusion_matrix_chart(metrics["confusion_matrix"]),
            use_container_width=True,
        )
    with right:
        st.markdown(
            '<div class="chart-header"><span class="chart-header-icon">📉</span><span class="chart-header-text">ROC Curve</span></div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            roc_curve_chart(metrics["fpr"], metrics["tpr"], metrics["auc"]),
            use_container_width=True,
        )

    st.markdown("---")

    # Classification report table
    st.markdown(
        '<div class="chart-header"><span class="chart-header-icon">📝</span><span class="chart-header-text">Classification Report</span></div>',
        unsafe_allow_html=True,
    )
    report_df = pd.DataFrame(report).T
    report_df = report_df.loc[["Ham", "Spam", "macro avg", "weighted avg"]]
    report_df["support"] = report_df["support"].astype(int)
    for c in ["precision", "recall", "f1-score"]:
        report_df[c] = report_df[c].map(lambda x: f"{x:.4f}")
    st.dataframe(report_df, use_container_width=True)

    st.markdown("---")

    # AUC info
    st.success(
        f"🏆  **ROC-AUC Score: {metrics['auc']:.4f}**  —  The model demonstrates excellent discriminative ability between spam and ham messages."
    )
