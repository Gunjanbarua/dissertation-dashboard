"""
proj3_plots.py
Plotly chart functions for Project 3 — Remote Sensing Integration.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

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
        font_size=12,
        font_family="Georgia, serif",
        font_color="#1A1A1A",
        bordercolor=_COLORS["sage"],
    ),
)

# Human-readable labels for prediction columns
_MODEL_LABELS: dict[str, str] = {
    "pred_GRU":      "GRU",
    "pred_RF":       "Random Forest",
    "pred_LSTM":     "LSTM",
    "pred_XGBoost":  "XGBoost",
    "pred_LightGBM": "LightGBM",
    "pred_SVR":      "SVR",
    "pred_GBM":      "GBM",
    # reduced-set variants (proj3_reduced.csv)
    "pred_GRUreduced":  "GRU (Reduced)",
    "pred_RFreduced":   "RF (Reduced)",
    "pred_LSTMreduced": "LSTM (Reduced)",
}

# Colour per thinning status — Viridis endpoints for maximum contrast
_THINNING_PALETTE: dict = {
    "thinned":   "#1f9e89",   # Viridis teal
    "unthinned": "#440154",   # Viridis dark purple
    "Thinned":   "#1f9e89",
    "Unthinned": "#440154",
    True:        "#1f9e89",
    False:       "#440154",
    1:           "#1f9e89",
    0:           "#440154",
}


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def plot_p3_scatter(df: pd.DataFrame) -> go.Figure:
    """
    Scatter: observed vs. predicted stand volume for all models.
    Tab buttons switch between models; R² shown top-left corner.

    Parameters
    ----------
    df : pd.DataFrame  (proj3.csv)

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if df.empty:
        return _empty_figure("No data available — place proj3.csv in data/.")

    # Fixed display order
    _ORDERED = [
        ("pred_RF",       "Random Forest"),
        ("pred_GRU",      "GRU"),
        ("pred_LSTM",     "LSTM"),
        ("pred_XGBoost",  "XGBoost"),
        ("pred_LightGBM", "LightGBM"),
        ("pred_SVR",      "SVR"),
        ("pred_GBM",      "GBM"),
    ]
    model_cols = [(col, lbl) for col, lbl in _ORDERED if col in df.columns]
    if not model_cols:
        return _empty_figure("No prediction columns found in proj3.csv.")

    df = df.copy().dropna(subset=["Vol_m3"])
    thinning_vals = df["thinning_status"].dropna().unique().tolist()
    n_models = len(model_cols)
    n_thin   = len(thinning_vals)

    # Global axis range across all models
    all_vals = pd.concat(
        [df["Vol_m3"]] + [df[c].dropna() for c, _ in model_cols]
    ).dropna()
    vmin, vmax = all_vals.min() * 0.94, all_vals.max() * 1.06

    # R² helper
    def _r2(col: str) -> float:
        sub = df.dropna(subset=[col])
        obs, pred = sub["Vol_m3"], sub[col]
        ss_res = ((obs - pred) ** 2).sum()
        ss_tot = ((obs - obs.mean()) ** 2).sum()
        return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Annotation builders
    def _r2_ann(r2_val: float) -> dict:
        return dict(
            x=0.03, y=0.97,
            xref="paper", yref="paper",
            text=f"<b>R² = {r2_val:.2f}</b>",
            showarrow=False,
            font=dict(size=14, color=_COLORS["dark"], family="Georgia, serif"),
            bgcolor="#F5F0E8",
            bordercolor=_COLORS["sage"],
            borderwidth=1,
            borderpad=5,
            align="left",
        )

    _one_to_one_ann = dict(
        x=vmax * 0.92, y=vmax * 0.85,
        xref="x", yref="y",
        text="1:1 Line",
        showarrow=False,
        font=dict(color=_COLORS["rust"], size=10, family="Georgia, serif"),
    )

    fig = go.Figure()

    # ── Legend-only traces (always visible, never in visibility toggle) ──────
    for status in thinning_vals:
        color = _THINNING_PALETTE.get(status, _COLORS["mist"])
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            name=str(status).title(),
            showlegend=True,
            legendgroup=str(status),
            marker=dict(color=color, size=9, opacity=0.78,
                        line=dict(width=0.8, color="#FFFFFF")),
        ))

    # ── Data traces (n_models × n_thin) ──────────────────────────────────────
    optional_cols = ["tph", "plot_ID", "area_ha"]
    for m_idx, (col, lbl) in enumerate(model_cols):
        sub_df = df.dropna(subset=[col])
        for status in thinning_vals:
            sub = sub_df[sub_df["thinning_status"] == status]
            color = _THINNING_PALETTE.get(status, _COLORS["mist"])
            available = [c for c in optional_cols if c in sub.columns]
            customdata = sub[available].values if available else None

            hover_parts = [
                "<b>True Vol.:</b> %{x:.2f} m³/ha",
                f"<b>Predicted ({lbl}):</b>" + " %{y:.2f} m³/ha",
            ]
            if "plot_ID" in available:
                hover_parts.insert(0, "<b>Plot ID:</b> %{customdata[1]}")
            if "tph" in available:
                hover_parts.append("<b>TPH:</b> %{customdata[0]:.0f}")
            if "area_ha" in available:
                hover_parts.append("<b>Area:</b> %{customdata[2]:.2f} ha")

            fig.add_trace(go.Scatter(
                x=sub["Vol_m3"],
                y=sub[col],
                mode="markers",
                name=str(status).title(),
                legendgroup=str(status),
                showlegend=False,
                visible=(m_idx == 0),
                marker=dict(color=color, size=9, opacity=0.78,
                            line=dict(width=0.8, color="#FFFFFF")),
                customdata=customdata,
                hovertemplate="<br>".join(hover_parts) + "<extra></extra>",
            ))

    # ── Tab buttons ───────────────────────────────────────────────────────────
    buttons = []
    for m_idx, (col, lbl) in enumerate(model_cols):
        r2_val = _r2(col)
        # legend traces always True; data traces toggle by model index
        vis = [True] * n_thin + [
            (mi == m_idx)
            for mi in range(n_models)
            for _ in range(n_thin)
        ]
        buttons.append(dict(
            label=lbl,
            method="update",
            args=[
                {"visible": vis},
                {
                    "title": f"Stand Volume: Observed vs. {lbl} Predictions",
                    "annotations": [_one_to_one_ann, _r2_ann(r2_val)],
                },
            ],
        ))

    # 1:1 reference line (shape)
    fig.add_shape(
        type="line",
        x0=vmin, y0=vmin, x1=vmax, y1=vmax,
        line=dict(color=_COLORS["rust"], dash="dash", width=1.8),
    )

    fig.update_layout(
        **_LAYOUT_BASE,
        title=f"Stand Volume: Observed vs. {model_cols[0][1]} Predictions",
        annotations=[_one_to_one_ann, _r2_ann(_r2(model_cols[0][0]))],
        xaxis=dict(
            title="Observed Volume (m\u00b3 ha\u207b\u00b9)",
            range=[vmin, vmax],
            gridcolor="#EAE6DE",
            zeroline=False,
            title_font=dict(color="#1A1A1A"),
            tickfont=dict(color="#1A1A1A"),
        ),
        yaxis=dict(
            title="Predicted Volume (m\u00b3 ha\u207b\u00b9)",
            range=[vmin, vmax],
            gridcolor="#EAE6DE",
            zeroline=False,
            title_font=dict(color="#1A1A1A"),
            tickfont=dict(color="#1A1A1A"),
        ),
        legend=dict(
            title="Thinning Status",
            bgcolor="#F5F0E8",
            bordercolor=_COLORS["sage"],
            borderwidth=1,
            font=dict(color="#1A1A1A"),
            title_font=dict(color="#1A1A1A"),
        ),
        updatemenus=[dict(
            type="buttons",
            direction="right",
            active=0,
            buttons=buttons,
            x=0.0, xanchor="left",
            y=1.13, yanchor="top",
            bgcolor="#F5F0E8",
            bordercolor=_COLORS["sage"],
            font=dict(size=11, color="#1A1A1A"),
        )],
    )
    return fig


def plot_p3_rmse_by_tph(df: pd.DataFrame) -> go.Figure:
    """
    2×4 subplot grid: RMSE by TPH density category, one panel per model.
    Mirrors the style of figure4_rmse_tph_category in the paper.

    Parameters
    ----------
    df : pd.DataFrame  (proj3.csv)

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if df.empty:
        return _empty_figure("No data available.")

    _ORDERED = [
        ("pred_GRU",      "GRU"),
        ("pred_RF",       "RF"),
        ("pred_LSTM",     "LSTM"),
        ("pred_XGBoost",  "XGBoost"),
        ("pred_LightGBM", "LightGBM"),
        ("pred_SVR",      "SVR"),
        ("pred_GBM",      "GBM"),
    ]
    model_cols = [(col, lbl) for col, lbl in _ORDERED if col in df.columns]
    if not model_cols:
        return _empty_figure("No prediction columns found in proj3.csv.")

    df = df.copy().dropna(subset=["Vol_m3", "tph"])

    # Density categories — right=False: [0,618) / [618,1237) / [1237,∞)
    cat_labels = ["Below 618", "618 to 1236", "1237 to 1853"]
    df["tph_cat"] = pd.cut(
        df["tph"],
        bins=[0, 618, 1237, float("inf")],
        labels=cat_labels,
        right=False,
    )

    # Viridis-derived palette for the three density groups
    _CAT_COLORS = {
        "Below 618":    "#26828e",   # Viridis teal-blue
        "618 to 1236":  "#3e4989",   # Viridis indigo
        "1237 to 1853": "#6ece58",   # Viridis light green
    }

    def _rmse(obs: pd.Series, pred: pd.Series) -> float:
        if len(obs) == 0:
            return 0.0
        return float(np.sqrt(((obs - pred) ** 2).mean()))

    # Pre-compute all RMSE values
    rmse_data: dict = {}
    for col, _ in model_cols:
        sub = df.dropna(subset=[col])
        rmse_data[col] = {
            cat: _rmse(
                sub.loc[sub["tph_cat"] == cat, "Vol_m3"],
                sub.loc[sub["tph_cat"] == cat, col],
            )
            for cat in cat_labels
        }

    # Shared y-axis ceiling across all panels for fair comparison
    global_max = max(v for d in rmse_data.values() for v in d.values())
    y_top = global_max * 1.28

    # Build 2×4 subplot grid
    n_models = len(model_cols)
    ncols, nrows = 4, 2
    titles = [lbl for _, lbl in model_cols]
    while len(titles) < nrows * ncols:
        titles.append("")

    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=titles,
        vertical_spacing=0.22,
        horizontal_spacing=0.07,
    )

    legend_added: set = set()
    for m_idx, (col, lbl) in enumerate(model_cols):
        r, c = m_idx // ncols + 1, m_idx % ncols + 1
        for cat in cat_labels:
            rmse_val = rmse_data[col][cat]
            show_leg = cat not in legend_added
            legend_added.add(cat)
            fig.add_trace(
                go.Bar(
                    name=cat,
                    x=[cat],
                    y=[rmse_val],
                    text=[f"{rmse_val:.2f}"],
                    textposition="outside",
                    textfont=dict(size=10, color="#1A1A1A", family="Georgia, serif"),
                    marker_color=_CAT_COLORS[cat],
                    marker_opacity=0.85,
                    marker_line=dict(width=0.4, color="#FFFFFF"),
                    showlegend=show_leg,
                    legendgroup=cat,
                    hovertemplate=(
                        f"<b>Model:</b> {lbl}<br>"
                        f"<b>Density:</b> {cat} trees ha\u207b\u00b9<br>"
                        "<b>RMSE:</b> %{y:.2f} m\u00b3 ha\u207b\u00b9"
                        "<extra></extra>"
                    ),
                ),
                row=r, col=c,
            )

    # Global axis styling
    fig.update_yaxes(
        range=[0, y_top],
        gridcolor="#EAE6DE",
        zeroline=False,
        tickfont=dict(color="#1A1A1A", family="Georgia, serif", size=10),
    )
    fig.update_xaxes(
        showticklabels=False,
        zeroline=False,
        gridcolor="rgba(0,0,0,0)",
    )
    # Y-axis title for leftmost column panels only
    for row_i in range(1, nrows + 1):
        fig.update_yaxes(
            title_text="RMSE (m\u00b3 ha\u207b\u00b9)",
            title_font=dict(color="#1A1A1A", family="Georgia, serif", size=11),
            row=row_i, col=1,
        )

    fig.update_layout(
        **_LAYOUT_BASE,
        title="RMSE by Stand Density Category",
        height=520,
        showlegend=True,
        bargap=0.30,
        legend=dict(
            title="Density (trees ha\u207b\u00b9)",
            bgcolor="#F5F0E8",
            bordercolor=_COLORS["sage"],
            borderwidth=1,
            font=dict(color="#1A1A1A", family="Georgia, serif"),
            title_font=dict(color="#1A1A1A"),
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="center",
            x=0.5,
        ),
    )
    # Separate call to safely override margin (avoids duplicate-key bug with _LAYOUT_BASE)
    fig.update_layout(margin=dict(l=70, r=50, t=115, b=60))

    return fig


def plot_p3_importance(df: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar chart of remote-sensing feature importance.
    Bars sorted highest → lowest; tab buttons switch between models.

    Parameters
    ----------
    df : pd.DataFrame  (proj3_importance.csv)

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if df.empty:
        return _empty_figure("No data available — place proj3_importance.csv in data/.")

    df = df.copy().dropna(subset=["feature", "importance_mean", "model"])
    models = df["model"].dropna().unique().tolist()
    n_models = len(models)

    fig = go.Figure()

    for i, model in enumerate(models):
        # autorange="reversed" flips the y-axis so the first item in the list
        # appears at the top → sort descending to put highest bar at top
        sub = (
            df[df["model"] == model]
            .sort_values("importance_mean", ascending=False)
        )
        fig.add_trace(go.Bar(
            name=model,
            x=sub["importance_mean"],
            y=sub["feature"],
            orientation="h",
            marker=dict(
                color=_COLOR_SEQ[i % len(_COLOR_SEQ)],
                opacity=0.85,
                line=dict(width=0.4, color="#FFFFFF"),
            ),
            hovertemplate=(
                "<b>Feature:</b> %{y}<br>"
                f"<b>Model:</b> {model}<br>"
                "<b>Mean Importance:</b> %{x:.4f}"
                "<extra></extra>"
            ),
            visible=(i == 0),
        ))

    buttons = [
        dict(
            label=model,
            method="update",
            args=[
                {"visible": [j == i for j in range(n_models)]},
                {"title": f"Remote Sensing Feature Importance — {model}"},
            ],
        )
        for i, model in enumerate(models)
    ]

    fig.update_layout(
        **_LAYOUT_BASE,
        title=f"Remote Sensing Feature Importance — {models[0]}",
        xaxis=dict(
            title="Mean Importance Score (Increase in RMSE)",
            gridcolor="#EAE6DE",
            zeroline=False,
            title_font=dict(color="#1A1A1A"),
            tickfont=dict(color="#1A1A1A"),
        ),
        yaxis=dict(
            title="",
            autorange="reversed",
            gridcolor="rgba(0,0,0,0)",
            tickfont=dict(color="#1A1A1A"),
        ),
        showlegend=False,
        bargap=0.28,
        updatemenus=[dict(
            type="buttons",
            direction="right",
            active=0,
            buttons=buttons,
            x=0.0, xanchor="left",
            y=1.13, yanchor="top",
            bgcolor="#F5F0E8",
            bordercolor=_COLORS["sage"],
            font=dict(size=11, color="#1A1A1A"),
        )],
    )
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
