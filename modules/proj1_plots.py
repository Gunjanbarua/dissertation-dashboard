"""
proj1_plots.py
Plotly chart functions for Project 1 — Individual Tree Volume Prediction.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Shared theme constants
# ---------------------------------------------------------------------------
_COLORS = {
    "sage":  "#7A9E7E",
    "teal":  "#5B7B7A",
    "beige": "#C5B89A",
    "dark":  "#3D5A5A",
    "rust":  "#A67C6A",
    "mist":  "#8B9E9E",
    "straw": "#C8B87A",
}

_COLOR_SEQ = [
    "#440154",  # Viridis 0 — dark purple
    "#3e4989",  # Viridis 2 — indigo
    "#31688e",  # Viridis 3 — blue
    "#26828e",  # Viridis 4 — teal-blue
    "#1f9e89",  # Viridis 5 — teal
    "#35b779",  # Viridis 6 — green
    "#6ece58",  # Viridis 7 — light green
    "#b5de2b",  # Viridis 8 — yellow-green
]

_LAYOUT_BASE = dict(
    template="plotly_white",
    font=dict(family="Georgia, serif", size=12, color="#1A1A1A"),
    paper_bgcolor="#FAFAF7",
    plot_bgcolor="#FAFAF7",
    margin=dict(l=70, r=50, t=70, b=60),
    hoverlabel=dict(
        bgcolor="#F5F0E8",
        font_color="#1A1A1A",
        font_size=12,
        font_family="Georgia, serif",
        bordercolor=_COLORS["sage"],
    ),
)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def plot_p1_scatter(df: pd.DataFrame) -> go.Figure:
    """
    Scatter plot of observed vs. predicted individual tree volume.
    One trace per model; tab buttons switch between models.
    R² updates per model in the top-left annotation.
    """
    if df.empty:
        return _empty_figure("No data available — place proj1.csv in data/.")

    df = df.copy().dropna(subset=["volume_m_21", "prediction_m"])
    models = sorted(df["model"].dropna().unique().tolist())

    # Per-model R²
    r2_values = {}
    for model in models:
        sub = df[df["model"] == model]
        ss_res = ((sub["prediction_m"] - sub["volume_m_21"]) ** 2).sum()
        ss_tot = ((sub["volume_m_21"] - sub["volume_m_21"].mean()) ** 2).sum()
        r2_values[model] = (1 - ss_res / ss_tot) if ss_tot != 0 else 0.0

    # Stable axis range across all models
    all_vals = pd.concat([df["volume_m_21"], df["prediction_m"]]).dropna()
    vmin, vmax = all_vals.min() * 0.97, all_vals.max() * 1.03

    fig = go.Figure()

    custom_cols = ["treeID", "study", "tph", "reps"]
    for i, model in enumerate(models):
        sub = df[df["model"] == model]
        available = [c for c in custom_cols if c in sub.columns]
        customdata = sub[available].values if available else None

        fig.add_trace(go.Scatter(
            x=sub["volume_m_21"],
            y=sub["prediction_m"],
            mode="markers",
            name=model,
            marker=dict(
                color=_COLOR_SEQ[i % len(_COLOR_SEQ)],
                size=7,
                opacity=0.72,
                line=dict(width=0.5, color="#FFFFFF"),
            ),
            customdata=customdata,
            hovertemplate=(
                "<b>Tree ID:</b> %{customdata[0]}<br>"
                "<b>Study:</b> %{customdata[1]}<br>"
                "<b>Observed Vol.:</b> %{x:.3f} m\u00b3<br>"
                "<b>Predicted Vol.:</b> %{y:.3f} m\u00b3<br>"
                "<b>TPH:</b> %{customdata[2]}<br>"
                "<b>Rep:</b> %{customdata[3]}"
                "<extra></extra>"
            ),
            visible=(i == 0),
        ))

    # 1:1 reference line
    fig.add_shape(
        type="line",
        x0=vmin, y0=vmin, x1=vmax, y1=vmax,
        line=dict(color=_COLORS["rust"], dash="dash", width=1.8),
    )

    # Annotation dicts reused across buttons
    line_ann = dict(
        x=vmax * 0.94, y=vmax * 0.87,
        xref="x", yref="y",
        text="1:1 Line",
        showarrow=False,
        font=dict(color=_COLORS["rust"], size=10, family="Georgia, serif"),
    )

    def r2_ann(model):
        return dict(
            xref="paper", yref="paper",
            x=0.02, y=0.97,
            xanchor="left", yanchor="top",
            text=f"R\u00b2 = {r2_values[model]:.2f}",
            showarrow=False,
            font=dict(size=18, color="#1A1A1A", family="Georgia, serif"),
            bgcolor="#F5F0E8",
            bordercolor=_COLORS["sage"],
            borderpad=6,
        )

    # Set initial annotations
    fig.update_layout(annotations=[line_ann, r2_ann(models[0])])

    # Tab buttons
    buttons = []
    for i, model in enumerate(models):
        buttons.append(dict(
            label=model,
            method="update",
            args=[
                {"visible": [j == i for j in range(len(models))]},
                {
                    "title": f"Individual Tree Volume — {model}",
                    "annotations": [line_ann, r2_ann(model)],
                },
            ],
        ))

    fig.update_layout(
        **_LAYOUT_BASE,
        title=f"Individual Tree Volume — {models[0]}",
        showlegend=False,
        xaxis=dict(
            title="Observed Volume (m\u00b3 tree\u207b\u00b9)",
            range=[vmin, vmax],
            gridcolor="#EAE6DE",
            zeroline=False,
            title_font=dict(color="#1A1A1A"),
            tickfont=dict(color="#1A1A1A"),
        ),
        yaxis=dict(
            title="Predicted Volume (m\u00b3 tree\u207b\u00b9)",
            range=[vmin, vmax],
            gridcolor="#EAE6DE",
            zeroline=False,
            title_font=dict(color="#1A1A1A"),
            tickfont=dict(color="#1A1A1A"),
        ),
        updatemenus=[dict(
            type="buttons",
            direction="right",
            active=0,
            showactive=True,
            buttons=buttons,
            x=0.0, xanchor="left",
            y=1.15, yanchor="top",
            bgcolor="#F5F0E8",
            bordercolor=_COLORS["sage"],
            font=dict(size=14, color="#1A1A1A", family="Georgia, serif"),
        )],
    )
    fig.update_layout(margin=dict(l=70, r=30, t=110, b=60))
    return fig


def plot_p1_importance(df: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar chart of top-10 variable importance by model.

    y-axis   : Variable   — top-10 predictor names (most important at top)
    x-axis   : Importance — % increase in MSE
    selector : Model      — tab buttons (top-left, below title)
    """
    if df.empty:
        return _empty_figure("No data available — place proj1_importance.csv in data/.")

    df = df.copy().dropna(subset=["Variable", "Importance", "Model"])
    models = df["Model"].unique().tolist()

    fig = go.Figure()

    for i, model in enumerate(models):
        sub = (
            df[df["Model"] == model]
            .sort_values("Importance", ascending=False)
            .head(10)
            .sort_values("Importance", ascending=True)
        )
        fig.add_trace(go.Bar(
            name=model,
            x=sub["Importance"],
            y=sub["Variable"],
            orientation="h",
            marker=dict(
                color=_COLOR_SEQ[i % len(_COLOR_SEQ)],
                opacity=0.85,
                line=dict(width=0.4, color="#FFFFFF"),
            ),
            hovertemplate=(
                "<b>Variable:</b> %{y}<br>"
                f"<b>Model:</b> {model}<br>"
                "<b>% Increase in MSE:</b> %{x:.2f}%"
                "<extra></extra>"
            ),
            visible=(i == 0),
        ))

    buttons = []
    for i, model in enumerate(models):
        buttons.append(dict(
            label=model,
            method="update",
            args=[
                {"visible": [j == i for j in range(len(models))]},
                {"title": f"Feature Importance — {model}"},
            ],
        ))

    fig.update_layout(
        **_LAYOUT_BASE,
        title=f"Feature Importance — {models[0]}",
        xaxis=dict(
            title="Importance Score (Percentage increase in mean squared error)",
            gridcolor="#EAE6DE",
            zeroline=False,
            title_font=dict(color="#1A1A1A"),
            tickfont=dict(color="#1A1A1A"),
        ),
        yaxis=dict(
            title="",
            gridcolor="rgba(0,0,0,0)",
            tickfont=dict(color="#1A1A1A"),
        ),
        showlegend=False,
        bargap=0.28,
        updatemenus=[dict(
            type="buttons",
            direction="right",
            active=0,
            showactive=True,
            buttons=buttons,
            x=0.0, xanchor="left",
            y=1.15, yanchor="top",
            bgcolor="#F5F0E8",
            bordercolor=_COLORS["sage"],
            font=dict(size=14, color="#1A1A1A", family="Georgia, serif"),
        )],
    )
    fig.update_layout(margin=dict(l=70, r=50, t=110, b=60))
    return fig


def plot_p1_scatter_tph(df: pd.DataFrame) -> go.Figure:
    """
    Scatter subplots of observed vs. predicted individual tree volume,
    one subplot per unique TPH (stand density) treatment level.
    Tab buttons switch between models; each subplot shows its own R² value.
    """
    if df.empty:
        return _empty_figure("No data available — place proj1.csv in data/.")

    df = df.copy().dropna(subset=["volume_m_21", "prediction_m", "tph"])
    models = sorted(df["model"].dropna().unique().tolist())

    # Sort: numeric TPH values first (ascending), non-numeric (Nelder) last
    def _tph_key(v):
        try:
            return (0, int(str(v)))
        except ValueError:
            return (1, str(v))

    tph_vals = sorted(df["tph"].dropna().unique().tolist(), key=_tph_key)
    n_tph = len(tph_vals)
    n_models = len(models)

    # Stable axis range across all subplots
    all_vals = pd.concat([df["volume_m_21"], df["prediction_m"]]).dropna()
    vmin, vmax = all_vals.min() * 0.97, all_vals.max() * 1.03

    # Pre-compute R² per (model, tph)
    r2_map: dict = {}
    for model in models:
        for tph in tph_vals:
            sub = df[(df["model"] == model) & (df["tph"] == tph)]
            if len(sub) > 1:
                ss_res = ((sub["prediction_m"] - sub["volume_m_21"]) ** 2).sum()
                ss_tot = ((sub["volume_m_21"] - sub["volume_m_21"].mean()) ** 2).sum()
                r2_map[(model, tph)] = (1 - ss_res / ss_tot) if ss_tot != 0 else 0.0
            else:
                r2_map[(model, tph)] = float("nan")

    def _subplot_title(v):
        try:
            int(str(v))
            return str(v) + " Trees Per Hectare"
        except ValueError:
            return str(v)

    subplot_titles = [_subplot_title(t) for t in tph_vals]
    fig = make_subplots(
        rows=1, cols=n_tph,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.09,
    )

    # Capture subplot-title annotations before adding our own
    # Reduce font size so titles don't overlap on narrow/mobile screens
    subplot_title_anns = list(fig.layout.annotations)
    for ann in subplot_title_anns:
        ann["font"] = dict(size=10, family="Georgia, serif", color="#1A1A1A")

    custom_cols = ["treeID", "study", "reps"]
    for i, model in enumerate(models):
        model_df = df[df["model"] == model]
        for j, tph_val in enumerate(tph_vals):
            sub = model_df[model_df["tph"] == tph_val]
            available = [c for c in custom_cols if c in sub.columns]
            customdata = sub[available].values if available else None
            fig.add_trace(
                go.Scatter(
                    x=sub["volume_m_21"],
                    y=sub["prediction_m"],
                    mode="markers",
                    name=model,
                    showlegend=False,
                    marker=dict(
                        color=_COLOR_SEQ[i % len(_COLOR_SEQ)],
                        size=6,
                        opacity=0.72,
                        line=dict(width=0.5, color="#FFFFFF"),
                    ),
                    customdata=customdata,
                    hovertemplate=(
                        "<b>Tree ID:</b> %{customdata[0]}<br>"
                        "<b>Study:</b> %{customdata[1]}<br>"
                        "<b>Observed Vol.:</b> %{x:.3f} m<sup>3</sup><br>"
                        "<b>Predicted Vol.:</b> %{y:.3f} m<sup>3</sup>"
                        "<extra></extra>"
                    ),
                    visible=(i == 0),
                ),
                row=1, col=j + 1,
            )

    # 1:1 reference lines
    for j in range(n_tph):
        xref = "x" if j == 0 else "x" + str(j + 1)
        yref = "y" if j == 0 else "y" + str(j + 1)
        fig.add_shape(
            type="line",
            x0=vmin, y0=vmin, x1=vmax, y1=vmax,
            line=dict(color=_COLORS["rust"], dash="dash", width=1.5),
            xref=xref, yref=yref,
        )

    # R² annotation builder — one box per subplot
    def _r2_anns(model: str) -> list:
        anns = []
        for j, tph_val in enumerate(tph_vals):
            r2 = r2_map.get((model, tph_val), float("nan"))
            import math
            r2_text = f"R\u00b2 = {r2:.2f}" if not math.isnan(r2) else "R\u00b2 = N/A"
            xref = "x domain" if j == 0 else f"x{j + 1} domain"
            yref = "y domain" if j == 0 else f"y{j + 1} domain"
            anns.append(dict(
                xref=xref, yref=yref,
                x=0.03, y=0.97,
                xanchor="left", yanchor="top",
                text=r2_text,
                showarrow=False,
                font=dict(size=13, color="#1A1A1A", family="Georgia, serif"),
                bgcolor="#F5F0E8",
                bordercolor=_COLORS["sage"],
                borderpad=4,
            ))
        return anns

    # Tab buttons — toggle visibility and update R² annotations
    buttons = []
    for i, model in enumerate(models):
        vis = [idx // n_tph == i for idx in range(n_models * n_tph)]
        buttons.append(dict(
            label=model,
            method="update",
            args=[
                {"visible": vis},
                {
                    "title": "Individual Tree Volume by Stand Density \u2014 " + model,
                    "annotations": subplot_title_anns + _r2_anns(model),
                },
            ],
        ))

    fig.update_layout(
        **_LAYOUT_BASE,
        title="Individual Tree Volume by Stand Density \u2014 " + models[0],
        annotations=subplot_title_anns + _r2_anns(models[0]),
        showlegend=False,
        updatemenus=[dict(
            type="buttons",
            direction="right",
            active=0,
            showactive=True,
            buttons=buttons,
            x=0.0, xanchor="left",
            y=1.28, yanchor="top",
            bgcolor="#F5F0E8",
            bordercolor=_COLORS["sage"],
            font=dict(size=14, color="#1A1A1A", family="Georgia, serif"),
        )],
    )

    # Axis config — use title=dict(text,font) to ensure label color applies
    ax_style = dict(
        range=[vmin, vmax],
        gridcolor="#EAE6DE",
        zeroline=False,
        tickfont=dict(color="#1A1A1A"),
    )
    _tfont = dict(color="#1A1A1A", family="Georgia, serif", size=11)
    x_title = "Observed Volume (m\u00b3 tree\u207b\u00b9)"
    y_title = "Predicted Volume (m\u00b3 tree\u207b\u00b9)"

    fig.update_layout(
        xaxis=dict(ax_style, title=dict(text=x_title, font=_tfont)),
        yaxis=dict(ax_style, title=dict(text=y_title, font=_tfont)),
    )
    for j in range(1, n_tph):
        fig.update_layout(**{
            "xaxis" + str(j + 1): dict(ax_style, title=dict(text=x_title, font=_tfont)),
            "yaxis" + str(j + 1): dict(ax_style, title=dict(text="", font=_tfont)),
        })

    fig.update_layout(margin=dict(l=70, r=30, t=180, b=60))
    return fig


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=13, color="#1A1A1A", family="Georgia, serif"),
    )
    fig.update_layout(**_LAYOUT_BASE)
    return fig
