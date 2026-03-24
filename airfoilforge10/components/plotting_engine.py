"""
Plotly-based plotting engine — replaces all matplotlib in the original app.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

COLORS = px.colors.qualitative.Plotly

DARK_LAYOUT = dict(
    paper_bgcolor="#1e2128",
    plot_bgcolor="#0f1117",
    font=dict(color="#e0e0e0", family="Segoe UI, Inter, sans-serif"),
    legend=dict(bgcolor="rgba(30,33,40,0.8)", bordercolor="#444", borderwidth=1),
    margin=dict(l=50, r=20, t=40, b=50),
)

PLOT_MAP = {
    "CL vs CD":            ("CD",    "CL",                   "Polar Curve",           "CD",        "CL"),
    "CL vs Alpha":         ("alpha", "CL",                   "Lift Curve",            "Alpha (°)", "CL"),
    "CL/CD vs Alpha":      ("alpha", "CL/CD",                "Efficiency Curve",      "Alpha (°)", "CL/CD"),
    "CD vs Alpha":         ("alpha", "CD",                   "Drag Curve",            "Alpha (°)", "CD"),
    "Cm vs Alpha":         ("alpha", "CM",                   "Moment Curve",          "Alpha (°)", "Cm"),
    "Confidence vs Alpha": ("alpha", "analysis_confidence",  "NeuralFoil Confidence", "Alpha (°)", "Confidence"),
    "Cpmin vs Alpha":      ("alpha", "Cpmin",                "Min Pressure Coeff",    "Alpha (°)", "Cpmin"),
}


def make_polar_figure(aero_list, plot_types, names):
    """Return a Plotly figure with one subplot per plot type."""
    n = len(plot_types)
    if n == 0:
        return go.Figure()

    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    subplot_titles = [PLOT_MAP[pt][2] for pt in plot_types if pt in PLOT_MAP]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)

    for idx, pt in enumerate(plot_types):
        if pt not in PLOT_MAP:
            continue
        x_key, y_key, _, x_label, y_label = PLOT_MAP[pt]
        row = idx // cols + 1
        col = idx % cols + 1
        for j, (name, aero) in enumerate(zip(names, aero_list)):
            x = np.array(aero.get(x_key, []))
            y = np.array(aero.get(y_key, []))
            if x.size == 0 or y.size == 0:
                continue
            fig.add_trace(
                go.Scatter(
                    x=x, y=y,
                    mode="lines",
                    name=name,
                    line=dict(color=COLORS[j % len(COLORS)], width=2),
                    showlegend=(idx == 0),
                    hovertemplate=(
                        f"<b>{name}</b><br>"
                        f"{x_label}: %{{x:.3f}}<br>"
                        f"{y_label}: %{{y:.4f}}<extra></extra>"
                    ),
                ),
                row=row, col=col,
            )
        fig.update_xaxes(title_text=x_label, row=row, col=col,
                         gridcolor="#333", zerolinecolor="#555")
        fig.update_yaxes(title_text=y_label, row=row, col=col,
                         gridcolor="#333", zerolinecolor="#555")

    fig.update_layout(
        **DARK_LAYOUT,
        height=350 * rows,
        legend_tracegroupgap=4,
    )
    return fig


def make_shape_figure(coords_list, names=None, halo=False, pitch=0.0):
    """Return a Plotly figure of airfoil outlines."""
    fig = go.Figure()
    for i, coords in enumerate(coords_list):
        theta = np.deg2rad(pitch)
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta),  np.cos(theta)]])
        c = coords @ rot.T
        label = names[i] if names else f"Airfoil {i+1}"
        fig.add_trace(go.Scatter(
            x=c[:, 0], y=c[:, 1],
            mode="lines",
            name=label,
            line=dict(color=COLORS[i % len(COLORS)], width=2),
        ))
        if halo:
            dirs = np.roll(c, -1, axis=0) - c
            nlen = np.linalg.norm(dirs, axis=1)
            nlen[nlen == 0] = 1
            norms = np.column_stack((-dirs[:, 1], dirs[:, 0])) / nlen[:, np.newaxis]
            offset = 0.01 * (np.max(c[:, 1]) - np.min(c[:, 1]))
            hc = c + offset * norms
            fig.add_trace(go.Scatter(
                x=hc[:, 0], y=hc[:, 1],
                mode="lines",
                name=f"{label} halo",
                line=dict(color=COLORS[i % len(COLORS)], width=1, dash="dash"),
            ))

    layout = {**DARK_LAYOUT, "margin": dict(l=30, r=30, t=20, b=30)}
    fig.update_layout(
        **layout,
        yaxis_scaleanchor="x",
        height=300,
        xaxis=dict(gridcolor="#333"),
        yaxis=dict(gridcolor="#333"),
    )
    return fig


def make_reynolds_gauge(re_value):
    """Gauge chart showing Reynolds number and flow regime."""
    if re_value < 5e4:
        regime, color = "Very Low Re (creeping flow)", "#9b59b6"
    elif re_value < 5e5:
        regime, color = "Low Re (laminar-dominant)", "#3498db"
    elif re_value < 5e6:
        regime, color = "Moderate Re (transitional)", "#27ae60"
    else:
        regime, color = "High Re (turbulent-dominant)", "#e74c3c"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=re_value,
        number={"valueformat": ".3s", "font": {"size": 28, "color": "#fff"}},
        title={"text": f"Reynolds Number — {regime}", "font": {"size": 13, "color": "#aaa"}},
        gauge={
            "axis": {
                "range": [0, 1e7],
                "tickformat": ".2s",
                "tickfont": {"color": "#aaa"},
                "tickcolor": "#555",       # ← valid property (not gridcolor)
                "tickwidth": 1,
            },
            "bar": {"color": color},
            "bgcolor": "#1e2128",
            "bordercolor": "#444",
            "borderwidth": 1,
            "steps": [
                {"range": [0,    5e4],  "color": "#2c1654"},
                {"range": [5e4,  5e5],  "color": "#1a3a5e"},
                {"range": [5e5,  5e6],  "color": "#1a4030"},
                {"range": [5e6,  1e7],  "color": "#4a1a1a"},
            ],
        },
    ))
    fig.update_layout(
        **{**DARK_LAYOUT, "margin": dict(l=20, r=20, t=40, b=10)},
        height=220,
    )
    return fig
