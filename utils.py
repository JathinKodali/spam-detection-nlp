"""
Spam Message Detection — Visualization Utilities
=================================================
Premium Plotly charts and word cloud generation for the Streamlit dashboard.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud


# ──────────────────────────────────────────────
# Theme constants
# ──────────────────────────────────────────────
SPAM_COLOR = "#f43f5e"
SPAM_LIGHT = "#fb7185"
HAM_COLOR = "#10b981"
HAM_LIGHT = "#34d399"
INDIGO = "#6366f1"
INDIGO_LIGHT = "#818cf8"
BG_DARK = "rgba(0,0,0,0)"
CARD_BG = "rgba(255,255,255,0.02)"
GRID_COLOR = "rgba(255,255,255,0.04)"
TEXT_COLOR = "#e2e8f0"
TEXT_MUTED = "#64748b"
FONT_FAMILY = "Inter, system-ui, sans-serif"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=BG_DARK,
    plot_bgcolor=BG_DARK,
    font=dict(color=TEXT_COLOR, family=FONT_FAMILY, size=13),
    margin=dict(l=40, r=40, t=60, b=40),
    hoverlabel=dict(
        bgcolor="#1e293b",
        bordercolor="rgba(255,255,255,0.1)",
        font=dict(color=TEXT_COLOR, family=FONT_FAMILY),
    ),
)


def _apply_layout(fig, **kwargs):
    layout = {**PLOTLY_LAYOUT, **kwargs}
    fig.update_layout(**layout)
    fig.update_xaxes(
        gridcolor=GRID_COLOR,
        zeroline=False,
        title_font=dict(size=12, color=TEXT_MUTED),
        tickfont=dict(size=11, color=TEXT_MUTED),
    )
    fig.update_yaxes(
        gridcolor=GRID_COLOR,
        zeroline=False,
        title_font=dict(size=12, color=TEXT_MUTED),
        tickfont=dict(size=11, color=TEXT_MUTED),
    )
    return fig


# ──────────────────────────────────────────────
# Class distribution donut
# ──────────────────────────────────────────────
def class_distribution_chart(df: pd.DataFrame):
    counts = df["label"].value_counts().reset_index()
    counts.columns = ["Label", "Count"]
    fig = px.pie(
        counts,
        names="Label",
        values="Count",
        color="Label",
        color_discrete_map={"spam": SPAM_COLOR, "ham": HAM_COLOR},
        hole=0.62,
    )
    fig.update_traces(
        textinfo="label+percent",
        textfont=dict(size=15, family=FONT_FAMILY, color="white"),
        marker=dict(line=dict(color="#06080f", width=3)),
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}<extra></extra>",
        pull=[0.02, 0.02],
    )
    _apply_layout(
        fig,
        title=dict(
            text="Class Distribution",
            font=dict(size=18, color=TEXT_COLOR),
            x=0.5,
        ),
        height=400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=13),
        ),
    )
    # Add annotation in center
    fig.add_annotation(
        text=f"<b>{len(df):,}</b><br><span style='font-size:11px;color:{TEXT_MUTED}'>messages</span>",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=22, color=TEXT_COLOR, family=FONT_FAMILY),
    )
    return fig


# ──────────────────────────────────────────────
# Message length histogram
# ──────────────────────────────────────────────
def message_length_distribution(df: pd.DataFrame):
    tmp = df.copy()
    tmp["length"] = tmp["message"].str.len()
    fig = px.histogram(
        tmp,
        x="length",
        color="label",
        barmode="overlay",
        color_discrete_map={"spam": SPAM_COLOR, "ham": HAM_COLOR},
        nbins=60,
        opacity=0.65,
        labels={"length": "Message Length (characters)", "label": "Label", "count": "Count"},
    )
    fig.update_traces(
        hovertemplate="Length: %{x}<br>Count: %{y}<extra></extra>",
    )
    _apply_layout(
        fig,
        title=dict(text="Message Length Distribution", font=dict(size=18, color=TEXT_COLOR), x=0.5),
        height=420,
        bargap=0.05,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=12),
        ),
    )
    return fig


# ──────────────────────────────────────────────
# Word cloud
# ──────────────────────────────────────────────
def generate_wordcloud(texts, label="Spam"):
    text = " ".join(texts)
    base = SPAM_LIGHT if label == "Spam" else HAM_LIGHT

    def color_func(*args, **kwargs):
        # Slight color variation for visual interest
        import random
        if label == "Spam":
            colors = ["#f43f5e", "#fb7185", "#fda4af", "#e11d48", "#ff6b8a"]
        else:
            colors = ["#10b981", "#34d399", "#6ee7b7", "#059669", "#4ade80"]
        return random.choice(colors)

    wc = WordCloud(
        width=900,
        height=450,
        background_color="#0a0e1a",
        color_func=color_func,
        max_words=150,
        contour_color=base,
        contour_width=1,
        prefer_horizontal=0.8,
        relative_scaling=0.5,
        min_font_size=8,
        max_font_size=80,
    ).generate(text if text.strip() else "empty")
    return wc.to_image()


# ──────────────────────────────────────────────
# Confusion matrix heatmap
# ──────────────────────────────────────────────
def confusion_matrix_chart(cm):
    labels = ["Ham", "Spam"]
    # Normalize for color scale
    cm_norm = cm.astype(float)
    for i in range(len(cm)):
        row_sum = cm[i].sum()
        if row_sum > 0:
            cm_norm[i] = cm[i] / row_sum

    fig = go.Figure(
        data=go.Heatmap(
            z=cm_norm,
            x=labels,
            y=labels,
            text=[[str(v) for v in row] for row in cm],
            texttemplate="%{text}",
            textfont=dict(size=24, color="white", family=FONT_FAMILY),
            colorscale=[
                [0, "#0f172a"],
                [0.3, "#312e81"],
                [0.6, "#6366f1"],
                [1, "#a78bfa"],
            ],
            showscale=False,
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{text}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Predicted", side="bottom")
    fig.update_yaxes(title="Actual", autorange="reversed")
    _apply_layout(
        fig,
        title=dict(text="Confusion Matrix", font=dict(size=18, color=TEXT_COLOR), x=0.5),
        height=420,
        width=460,
    )
    return fig


# ──────────────────────────────────────────────
# ROC curve
# ──────────────────────────────────────────────
def roc_curve_chart(fpr, tpr, auc):
    fig = go.Figure()
    # AUC fill
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"ROC (AUC = {auc:.4f})",
            line=dict(color=INDIGO, width=3),
            fill="tozeroy",
            fillcolor="rgba(99, 102, 241, 0.08)",
            hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
        )
    )
    # Random baseline
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Baseline",
            line=dict(color="rgba(255,255,255,0.15)", width=1.5, dash="dash"),
            hoverinfo="skip",
        )
    )
    fig.update_xaxes(title="False Positive Rate", range=[0, 1])
    fig.update_yaxes(title="True Positive Rate", range=[0, 1.02])
    _apply_layout(
        fig,
        title=dict(text="ROC Curve", font=dict(size=18, color=TEXT_COLOR), x=0.5),
        height=420,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
    )
    # AUC annotation
    fig.add_annotation(
        x=0.6,
        y=0.3,
        text=f"<b>AUC = {auc:.4f}</b>",
        showarrow=False,
        font=dict(size=15, color=INDIGO_LIGHT, family=FONT_FAMILY),
        bgcolor="rgba(99,102,241,0.08)",
        bordercolor="rgba(99,102,241,0.2)",
        borderwidth=1,
        borderpad=8,
    )
    return fig


# ──────────────────────────────────────────────
# Top features
# ──────────────────────────────────────────────
def top_features_chart(top_spam_words, top_ham_words):
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Top Spam Indicators", "Top Ham Indicators"),
        horizontal_spacing=0.18,
    )
    fig.update_annotations(font=dict(size=14, color=TEXT_COLOR, family=FONT_FAMILY))

    n = 15
    spam_words = [w for w, _ in top_spam_words[:n]]
    spam_vals = [v for _, v in top_spam_words[:n]]
    ham_words = [w for w, _ in top_ham_words[:n]]
    ham_vals = [abs(v) for _, v in top_ham_words[:n]]

    fig.add_trace(
        go.Bar(
            y=spam_words[::-1],
            x=spam_vals[::-1],
            orientation="h",
            marker=dict(
                color=spam_vals[::-1],
                colorscale=[[0, "#831843"], [0.5, "#e11d48"], [1, "#fb7185"]],
                cornerradius=4,
            ),
            name="Spam",
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            y=ham_words[::-1],
            x=ham_vals[::-1],
            orientation="h",
            marker=dict(
                color=ham_vals[::-1],
                colorscale=[[0, "#064e3b"], [0.5, "#059669"], [1, "#34d399"]],
                cornerradius=4,
            ),
            name="Ham",
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    _apply_layout(
        fig,
        title=dict(text="", font=dict(size=1)),
        height=550,
        showlegend=False,
    )
    return fig


# ──────────────────────────────────────────────
# Confidence gauge
# ──────────────────────────────────────────────
def confidence_gauge(confidence, label):
    color = SPAM_COLOR if label == "Spam" else HAM_COLOR
    light = SPAM_LIGHT if label == "Spam" else HAM_LIGHT

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            number=dict(
                suffix="%",
                font=dict(size=48, color=light, family=FONT_FAMILY),
            ),
            gauge=dict(
                axis=dict(
                    range=[0, 100],
                    tickwidth=1,
                    tickcolor="rgba(255,255,255,0.1)",
                    dtick=25,
                    tickfont=dict(size=10, color=TEXT_MUTED),
                ),
                bar=dict(color=color, thickness=0.75),
                bgcolor="rgba(255,255,255,0.03)",
                borderwidth=0,
                steps=[
                    dict(range=[0, 33], color="rgba(255,255,255,0.02)"),
                    dict(range=[33, 66], color="rgba(255,255,255,0.03)"),
                    dict(range=[66, 100], color="rgba(255,255,255,0.04)"),
                ],
                threshold=dict(
                    line=dict(color=light, width=3),
                    thickness=0.85,
                    value=confidence * 100,
                ),
            ),
            title=dict(
                text="Confidence",
                font=dict(size=14, color=TEXT_MUTED, family=FONT_FAMILY),
            ),
        )
    )
    _apply_layout(fig, height=280, margin=dict(l=30, r=30, t=50, b=10))
    return fig
