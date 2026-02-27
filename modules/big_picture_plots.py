"""
big_picture_plots.py
Plotly chart functions for The Big Picture synthesis page.
All data values are hardcoded from published dissertation results.
"""

import math
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Shared theme constants (match other plot modules exactly)
# ---------------------------------------------------------------------------
_COLORS = {
    "sage":  "#7A9E7E",
    "teal":  "#5B7B7A",
    "beige": "#C5B89A",
    "dark":  "#3D5A5A",
    "rust":  "#A67C6A",
    "mist":  "#8B9E9E",
}

_COLOR_SEQ = [
    "#440154",
    "#3e4989",
    "#31688e",
    "#26828e",
    "#1f9e89",
    "#35b779",
    "#6ece58",
    "#b5de2b",
]

_LAYOUT_BASE = dict(
    template="plotly_white",
    font=dict(family="Georgia, serif", size=12, color="#1A1A1A"),
    paper_bgcolor="#FAFAF7",
    plot_bgcolor="#FAFAF7",
    margin=dict(l=70, r=50, t=70, b=60),
    hoverlabel=dict(
        bgcolor="#F5F0E8",
        font_size=12,
        font_family="Georgia, serif",
        font_color="#1A1A1A",
        bordercolor=_COLORS["sage"],
    ),
)


# ---------------------------------------------------------------------------
# Chart 1: Scale Comparison — R² across all three projects
# ---------------------------------------------------------------------------

def plot_scale_comparison_metrics() -> go.Figure:
    """
    Horizontal bar chart comparing the best-model R² across all three
    project scales. Reference line at R² = 0.50.
    """
    projects    = [
        "🛰️  Landscape Scale · Satellite",
        "📈  Growth Over Time · Drone LiDAR",
        "🌳  Individual Tree · Drone LiDAR",
    ]
    r2_vals     = [0.49, 0.88, 0.59]
    best_models = ["GRU", "Random Forest", "SVR"]
    horizons    = ["Continuous", "7 years", "4 years"]
    colors      = [_COLORS["dark"], _COLORS["teal"], _COLORS["sage"]]

    fig = go.Figure()
    for proj, r2, model, horizon, color in zip(
        projects, r2_vals, best_models, horizons, colors
    ):
        fig.add_trace(go.Bar(
            name=proj,
            y=[proj],
            x=[r2],
            orientation="h",
            marker=dict(color=color, opacity=0.87,
                        line=dict(width=0.4, color="#FFFFFF")),
            text=[f"  R² = {r2:.2f}  |  Best: {model}  |  Horizon: {horizon}"],
            textposition="inside",
            textfont=dict(color="#FFFFFF", size=12, family="Georgia, serif"),
            hovertemplate=(
                f"<b>{proj}</b><br>"
                f"<b>Best Model:</b> {model}<br>"
                f"<b>R²:</b> {r2:.2f}<br>"
                f"<b>Prediction Horizon:</b> {horizon}"
                "<extra></extra>"
            ),
            showlegend=False,
        ))

    fig.add_vline(
        x=0.50,
        line=dict(color=_COLORS["rust"], dash="dash", width=1.8),
        annotation_text="Moderate Accuracy Threshold (R² = 0.50)",
        annotation_position="top right",
        annotation_font=dict(color=_COLORS["rust"], size=10,
                             family="Georgia, serif"),
    )

    fig.update_layout(
        **_LAYOUT_BASE,
        height=310,
        title="Best Predictive Accuracy (R²) by Spatial Scale",
        xaxis=dict(
            title="R² — Coefficient of Determination",
            range=[0, 1.08],
            gridcolor="#EAE6DE",
            zeroline=False,
            title_font=dict(color="#1A1A1A"),
            tickfont=dict(color="#1A1A1A"),
        ),
        yaxis=dict(
            title="",
            tickfont=dict(color="#1A1A1A", size=12),
            gridcolor="rgba(0,0,0,0)",
        ),
        bargap=0.38,
    )
    return fig


# ---------------------------------------------------------------------------
# Chart 2: Technology Pipeline — Sankey diagram
# ---------------------------------------------------------------------------

def plot_technology_pipeline() -> go.Figure:
    """
    Sankey diagram: sensor → feature type → ML model → prediction scale.
    """
    nodes = [
        # Sensors (0–2)
        "UAV-LiDAR",
        "Sentinel-1",
        "Sentinel-2",
        # Feature types (3–6)
        "3D Tree Struct.",
        "Canopy Comp.",
        "Radar Backsc.",
        "Spectral Refl.",
        # ML Models (7–9)
        "Random Forest",
        "SVR",
        "GRU/LSTM",
        # Outputs (10–11)
        "Ind. Tree Vol.",
        "Plot-Level Vol.",
    ]

    node_colors = (
        [_COLORS["dark"]]  * 3 +   # Sensors
        [_COLORS["teal"]]  * 4 +   # Feature types
        [_COLORS["sage"]]  * 3 +   # ML models
        [_COLORS["beige"]] * 2     # Outputs
    )

    # (source_idx, target_idx, flow_value)
    links = [
        (0, 3, 5),   # UAV-LiDAR → 3D Tree Structure
        (0, 4, 4),   # UAV-LiDAR → Canopy Competition
        (1, 5, 4),   # Sentinel-1 → Radar Backscatter
        (2, 6, 5),   # Sentinel-2 → Spectral Reflectance
        (3, 8, 3),   # 3D Tree Structure → SVR
        (3, 7, 3),   # 3D Tree Structure → Random Forest
        (4, 7, 4),   # Canopy Competition → Random Forest
        (4, 8, 3),   # Canopy Competition → SVR
        (5, 9, 4),   # Radar Backscatter → GRU/LSTM
        (6, 7, 2),   # Spectral Reflectance → Random Forest
        (6, 9, 3),   # Spectral Reflectance → GRU/LSTM
        (7, 10, 5),  # Random Forest → Individual Tree Volume
        (8, 10, 3),  # SVR → Individual Tree Volume
        (7, 11, 2),  # Random Forest → Plot-Level Volume
        (9, 11, 5),  # GRU/LSTM → Plot-Level Volume
    ]

    sources = [l[0] for l in links]
    targets = [l[1] for l in links]
    values  = [l[2] for l in links]

    _link_color_map = {
        0: "rgba(61,90,90,0.30)",
        1: "rgba(61,90,90,0.30)",
        2: "rgba(61,90,90,0.30)",
        3: "rgba(91,123,122,0.30)",
        4: "rgba(91,123,122,0.30)",
        5: "rgba(91,123,122,0.30)",
        6: "rgba(91,123,122,0.30)",
        7: "rgba(122,158,126,0.30)",
        8: "rgba(122,158,126,0.30)",
        9: "rgba(122,158,126,0.30)",
    }
    link_colors = [_link_color_map[s] for s in sources]

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20,
            thickness=22,
            line=dict(color="#FFFFFF", width=0.5),
            label=nodes,
            color=node_colors,
            hovertemplate="<b>%{label}</b><extra></extra>",
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
            hovertemplate=(
                "<b>%{source.label}</b> → <b>%{target.label}</b><br>"
                "<b>Relative Flow:</b> %{value}"
                "<extra></extra>"
            ),
        ),
    ))

    fig.update_layout(
        **_LAYOUT_BASE,
        height=420,
        title="From Sensor to Prediction: The Full Data Pipeline",
    )
    fig.update_layout(font=dict(family="Georgia, serif", size=9, color="#1A1A1A"))
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig


# ---------------------------------------------------------------------------
# Chart 3: Density vs. Accuracy
# ---------------------------------------------------------------------------

def plot_density_vs_accuracy() -> go.Figure:
    """
    Line chart showing how R² drops as stand density increases,
    one line per project.
    """
    density_labels = ["Low (618 TPH)", "Medium (1,236 TPH)", "High (1,853 TPH)"]

    project_lines = [
        dict(
            name="Project 1 · Individual Tree (SVR)",
            r2=[0.63, 0.52, 0.31],
            color=_COLORS["sage"],
        ),
        dict(
            name="Project 2 · Temporal Growth (RF avg.)",
            r2=[0.91, 0.83, 0.74],
            color=_COLORS["teal"],
        ),
        dict(
            name="Project 3 · Satellite (GRU)",
            r2=[0.55, 0.52, 0.35],
            color=_COLORS["dark"],
        ),
    ]

    fig = go.Figure()

    # Shaded "canopy closure zone" for the high-density category
    fig.add_vrect(
        x0=1.55, x1=2.55,
        fillcolor=_COLORS["rust"],
        opacity=0.07,
        line_width=0,
    )
    fig.add_annotation(
        x=2.0, y=0.97,
        text="⚠ Canopy interlocking<br>degrades sensor accuracy",
        showarrow=False,
        font=dict(color=_COLORS["rust"], size=10, family="Georgia, serif"),
        align="center",
        bgcolor="rgba(245,240,232,0.75)",
        bordercolor=_COLORS["rust"],
        borderwidth=1,
        borderpad=4,
    )

    for proj in project_lines:
        fig.add_trace(go.Scatter(
            x=density_labels,
            y=proj["r2"],
            mode="lines+markers",
            name=proj["name"],
            line=dict(color=proj["color"], width=2.4),
            marker=dict(size=12, color=proj["color"],
                        line=dict(width=1.5, color="#FFFFFF")),
            hovertemplate=(
                f"<b>{proj['name']}</b><br>"
                "<b>Density:</b> %{x}<br>"
                "<b>R²:</b> %{y:.2f}"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        **_LAYOUT_BASE,
        height=420,
        title="Stand Density vs. Model Accuracy Across All Three Projects",
        xaxis=dict(
            title="Stand Density (Trees per Hectare)",
            gridcolor="#EAE6DE",
            zeroline=False,
            title_font=dict(color="#1A1A1A"),
            tickfont=dict(color="#1A1A1A", size=10),
            tickangle=-30,
        ),
        yaxis=dict(
            title="R² (Model Accuracy)",
            range=[0.20, 1.02],
            gridcolor="#EAE6DE",
            zeroline=False,
            title_font=dict(color="#1A1A1A"),
            tickfont=dict(color="#1A1A1A"),
        ),
        legend=dict(
            bgcolor="#F5F0E8",
            bordercolor=_COLORS["sage"],
            borderwidth=1,
            font=dict(color="#1A1A1A", size=10),
            yanchor="top",
            y=-0.30,
            xanchor="left",
            x=0,
        ),
    )
    fig.update_layout(margin=dict(l=70, r=50, t=70, b=170))
    return fig


# ---------------------------------------------------------------------------
# Chart 4: Model Showdown — R² heatmap
# ---------------------------------------------------------------------------

def plot_model_showdown() -> go.Figure:
    """
    Heatmap of R² values across all models and all project scales.
    Cells are blank (NaN) where a model was not applied to a project.
    """
    models = ["Random Forest", "SVR", "GRU", "LSTM",
              "XGBoost", "LightGBM", "GBM"]
    project_labels = [
        "P3 · Satellite",
        "P2 · Temporal Growth",
        "P1 · Individual Tree",
    ]

    # R² matrix  [row = project, col = model]; None where not applicable
    z_raw = [
        [0.49, 0.40, 0.49, 0.49, 0.43, 0.46, 0.38],  # P3
        [0.85, 0.65, None, None, None, None, None],    # P2
        [0.47, 0.59, None, None, None, None, None],    # P1
    ]
    text = [
        ["0.49", "0.40", "0.49", "0.49", "0.43", "0.46", "0.38"],
        ["0.85", "0.65", "—",    "—",    "—",    "—",    "—"],
        ["0.47", "0.59", "—",    "—",    "—",    "—",    "—"],
    ]

    z_plot = [
        [v if v is not None else math.nan for v in row]
        for row in z_raw
    ]

    fig = go.Figure(go.Heatmap(
        z=z_plot,
        x=models,
        y=project_labels,
        text=text,
        texttemplate="%{text}",
        textfont=dict(color="#1A1A1A", size=9, family="Georgia, serif"),
        colorscale=[
            [0.0, "#F5F0E8"],   # low  → light beige
            [0.5, "#7A9E7E"],   # mid  → sage green
            [1.0, "#3D5A5A"],   # high → dark teal
        ],
        zmin=0.30,
        zmax=0.95,
        colorbar=dict(
            title="R²",
            title_font=dict(color="#1A1A1A", family="Georgia, serif", size=9),
            tickfont=dict(color="#1A1A1A", size=8),
            outlinecolor=_COLORS["sage"],
            outlinewidth=1,
            thickness=10,
        ),
        hovertemplate=(
            "<b>Model:</b> %{x}<br>"
            "<b>Project:</b> %{y}<br>"
            "<b>R²:</b> %{text}"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        **_LAYOUT_BASE,
        height=310,
        title="Model Performance Across All Three Projects (R²)",
        xaxis=dict(
            title="",
            tickfont=dict(color="#1A1A1A", size=9),
            tickangle=-30,
            gridcolor="rgba(0,0,0,0)",
        ),
        yaxis=dict(
            title="",
            tickfont=dict(color="#1A1A1A", size=9),
            gridcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_layout(margin=dict(l=110, r=20, t=70, b=100))
    return fig


# ---------------------------------------------------------------------------
# Chart 5: Feature Importance Synthesis — grouped bar chart
# ---------------------------------------------------------------------------

def plot_feature_importance_synthesis() -> go.Figure:
    """
    Grouped bar chart: approximate share (%) of predictive importance
    attributed to each broad feature category, broken down by project.
    Values are derived from published permutation importance results.
    """
    categories = [
        "Tree Structure",
        "Competition Indices",
        "SAR Radar Backscatter",
        "Optical / Spectral Bands",
    ]

    # Approximate % of top-feature importance by category
    project_data = [
        dict(
            name="P1 · Individual Tree (SVR)",
            values=[38, 52, 0, 10],
            color=_COLORS["sage"],
        ),
        dict(
            name="P2 · Temporal Growth (RF)",
            values=[28, 65, 0, 7],
            color=_COLORS["teal"],
        ),
        dict(
            name="P3 · Satellite (GRU)",
            values=[0, 0, 48, 52],
            color=_COLORS["dark"],
        ),
    ]

    fig = go.Figure()
    for proj in project_data:
        fig.add_trace(go.Bar(
            name=proj["name"],
            x=categories,
            y=proj["values"],
            marker=dict(color=proj["color"], opacity=0.87,
                        line=dict(width=0.4, color="#FFFFFF")),
            text=[f"{v}%" if v > 0 else "" for v in proj["values"]],
            textposition="outside",
            textfont=dict(size=11, color="#1A1A1A", family="Georgia, serif"),
            hovertemplate=(
                f"<b>{proj['name']}</b><br>"
                "<b>Feature Category:</b> %{x}<br>"
                "<b>Approx. Importance:</b> %{y}%"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        **_LAYOUT_BASE,
        height=430,
        title="What the Models Actually Learn: Feature Category Importance",
        barmode="group",
        bargap=0.20,
        bargroupgap=0.08,
        xaxis=dict(
            title="",
            tickfont=dict(color="#1A1A1A", size=10),
            gridcolor="rgba(0,0,0,0)",
            tickangle=-30,
        ),
        yaxis=dict(
            title="Approximate Share of Importance (%)",
            range=[0, 82],
            gridcolor="#EAE6DE",
            zeroline=False,
            title_font=dict(color="#1A1A1A"),
            tickfont=dict(color="#1A1A1A"),
        ),
        legend=dict(
            bgcolor="#F5F0E8",
            bordercolor=_COLORS["sage"],
            borderwidth=1,
            font=dict(color="#1A1A1A"),
        ),
    )
    return fig
