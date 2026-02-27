"""
proj2_plots.py
Plotly chart functions for Project 2 — Stand-Level Temporal Dynamics.
"""

import plotly.graph_objects as go
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
    margin=dict(l=70, r=50, t=130, b=80),
    hoverlabel=dict(
        bgcolor="#F5F0E8",
        font_color="#1A1A1A",
        font_size=12,
        font_family="Georgia, serif",
        bordercolor=_COLORS["sage"],
    ),
)

# Full-name labels for proj2 feature abbreviations (use <br> to wrap long entries)
_FEATURE_LABELS = {
    "Carea":      "Carea — Crown Area",
    "mCDst":      "mCDst — Mean Crown Distance",
    "Z":          "Z — Tree Top Height",
    "CArea_1":    "CArea_1 — Crown Area (Alt.)",
    "htlc":       "htlc — Height to Live Crown",
    "CLAI":       "CLAI — Crown Leaf Area Index",
    "vol1":       "vol1 — Volume (10th Pct.)",
    "vol2":       "vol2 — 3D convex hull volume<br>for lidar returns in the<br>top 20% of height",
    "vol3":       "vol3 — Volume (30th Pct.)",
    "vol4":       "vol4 — Volume (40th Pct.)",
    "vol5":       "vol5 — 3D convex hull volume<br>for lidar returns in the<br>top 50% of height",
    "sfa1":       "sfa1 — Surface area of 3D<br>convex hull for lidar returns<br>in the top 10% of height",
    "sfa2":       "sfa2 — Surface area of 3D<br>convex hull for lidar returns<br>in the top 20% of height",
    "sfa3":       "sfa3 — Surface Area (Top 30%)",
    "sfa4":       "sfa4 — Surface Area (Top 40%)",
    "sfa5":       "sfa5 — Surface area of 3D<br>convex hull for lidar returns<br>in the top 50% of height",
    "UndTF":      "UndTF — Understory Presence",
    "UndPrp":     "UndPrp — Understory Proportion",
    "CI_Carea":   "CI_Carea — Competition index<br>using areas derived from a<br>convex hull estimate of crown area",
    "CI_CArea_1": "CI_CArea_1 — CI: Crown Area (Alt.)",
    "CI_Z":       "CI_Z — Competition index<br>using the tree top height",
    "CI_mCDst":   "CI_mCDst — Competition index<br>using the maximum crown diameter",
    "CI_LAI":     "CI_LAI — CI: Leaf Area Index",
    "CI_HTLC":    "CI_HTLC — CI: Height to Crown",
    "CI_under":   "CI_under — CI: Understory",
    "CI_under2":  "CI_under2 — CI: Understory (Alt.)",
    "CI_vol1":    "CI_vol1 — CI: Volume (10th Pct.)",
    "CI_vol2":    "CI_vol2 — CI: Volume (20th Pct.)",
    "CI_vol3":    "CI_vol3 — CI: Volume (30th Pct.)",
    "CI_vol4":    "CI_vol4 — CI: Volume (40th Pct.)",
    "CI_vol5":    "CI_vol5 — CI: Volume (50th Pct.)",
    "CI_sfa1":    "CI_sfa1 — CI: Surface Area (10%)",
    "CI_sfa2":    "CI_sfa2 — CI: Surface Area (20%)",
    "CI_sfa3":    "CI_sfa3 — CI: Surface Area (30%)",
    "CI_sfa4":    "CI_sfa4 — CI: Surface Area (40%)",
    "CI_sfa5":    "CI_sfa5 — CI: Surface Area (50%)",
    "SILVA1":     "SILVA1 — Number of neighbors",
    "SILVA2":     "SILVA2 — Competition index<br>using number of neighbors",
    "age":        "age — Stand Age",
}


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

_SCATTER_SAMPLE = 400   # max points per (model, age) trace in animated scatter
_TPH_SAMPLE     = 200   # max points per (model, tph, age) trace in TPH subplot


def _sample(sub: pd.DataFrame, n: int) -> pd.DataFrame:
    """Return up to n rows, reproducibly sampled."""
    return sub.sample(n=min(n, len(sub)), random_state=42)


def plot_p2_scatter_animated(df: pd.DataFrame) -> go.Figure:
    """
    Scatter plot of observed vs. predicted stand volume with a Plotly
    age slider.  Each slider step filters to a single stand age and
    re-draws the scatter; Play/Pause buttons animate the progression.

    x-axis  : volume_m           — observed volume (m³)
    y-axis  : predicted_volume_m — model-predicted volume (m³)
    slider  : age                — stand age (years)
    colour  : model              — algorithm

    Parameters
    ----------
    df : pd.DataFrame  (proj2.csv)

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if df.empty:
        return _empty_figure("No data available — place proj2.csv in data/.")

    df = df.copy().dropna(subset=["volume_m", "predicted_volume_m", "age"])
    ages = sorted(df["age"].unique())
    models = sorted(df["model"].dropna().unique().tolist())
    model_colors = {m: _COLOR_SEQ[i % len(_COLOR_SEQ)] for i, m in enumerate(models)}
    n_models = len(models)

    # Stable axis range across all ages
    all_vol = pd.concat([df["volume_m"], df["predicted_volume_m"]]).dropna()
    vmin, vmax = all_vol.min() * 0.94, all_vol.max() * 1.06

    # Pre-compute R² per (age, model)
    r2_per_age = {}
    for model in models:
        for age_val in ages:
            sub = df[(df["age"] == age_val) & (df["model"] == model)]
            if len(sub) > 1:
                ss_res = ((sub["predicted_volume_m"] - sub["volume_m"]) ** 2).sum()
                ss_tot = ((sub["volume_m"] - sub["volume_m"].mean()) ** 2).sum()
                r2_per_age[(age_val, model)] = (1 - ss_res / ss_tot) if ss_tot != 0 else 0.0
            else:
                r2_per_age[(age_val, model)] = float("nan")

    # R² text position in data coordinates (upper-left area of plot)
    txt_x = vmin + 0.05 * (vmax - vmin)
    txt_y = vmin + 0.72 * (vmax - vmin)

    def _r2_text(age_val, model):
        r2 = r2_per_age.get((age_val, model), float("nan"))
        return f"R\u00b2 = {r2:.2f}" if not pd.isna(r2) else "R\u00b2 = N/A"

    # ---- Build one frame per age ----------------------------------------
    frames = []
    for age_val in ages:
        age_df = df[df["age"] == age_val]
        frame_traces = []

        # Scatter traces (one per model) — sampled to keep payload small
        for model in models:
            sub = _sample(age_df[age_df["model"] == model], _SCATTER_SAMPLE)
            frame_traces.append(go.Scatter(
                x=sub["volume_m"],
                y=sub["predicted_volume_m"],
                mode="markers",
                name=model,
                marker=dict(
                    color=model_colors[model],
                    size=8,
                    opacity=0.74,
                    line=dict(width=0.5, color="#FFFFFF"),
                ),
                customdata=sub[["tph", "plot", "study_x"]].values
                    if all(c in sub.columns for c in ["tph", "plot", "study_x"])
                    else sub[["tph"]].values,
                hovertemplate=(
                    f"<b>Age:</b> {age_val} yrs<br>"
                    "<b>Observed Vol.:</b> %{x:.3f} m\u00b3<br>"
                    "<b>Predicted Vol.:</b> %{y:.3f} m\u00b3<br>"
                    f"<b>Model:</b> {model}<br>"
                    "<b>TPH:</b> %{customdata[0]:.0f}"
                    "<extra></extra>"
                ),
            ))

        # Text traces (R² label per model — updated each frame)
        for model in models:
            frame_traces.append(go.Scatter(
                x=[txt_x],
                y=[txt_y],
                mode="text",
                text=[_r2_text(age_val, model)],
                textfont=dict(size=18, color="#1A1A1A", family="Georgia, serif"),
                showlegend=False,
                hoverinfo="skip",
            ))

        frames.append(go.Frame(data=frame_traces, name=str(age_val)))

    # ---- Initial data (first age) ----------------------------------------
    initial_age = ages[0]
    first_df = df[df["age"] == initial_age]
    initial_traces = []

    # Scatter traces — sampled
    for i, model in enumerate(models):
        sub = _sample(first_df[first_df["model"] == model], _SCATTER_SAMPLE)
        initial_traces.append(go.Scatter(
            x=sub["volume_m"],
            y=sub["predicted_volume_m"],
            mode="markers",
            name=model,
            visible=(i == 0),
            marker=dict(
                color=model_colors[model],
                size=8,
                opacity=0.74,
                line=dict(width=0.5, color="#FFFFFF"),
            ),
            customdata=sub[["tph", "plot", "study_x"]].values
                if all(c in sub.columns for c in ["tph", "plot", "study_x"])
                else sub[["tph"]].values,
            hovertemplate=(
                f"<b>Age:</b> {initial_age} yrs<br>"
                "<b>Observed Vol.:</b> %{x:.3f} m\u00b3<br>"
                "<b>Predicted Vol.:</b> %{y:.3f} m\u00b3<br>"
                f"<b>Model:</b> {model}<br>"
                "<b>TPH:</b> %{customdata[0]:.0f}"
                "<extra></extra>"
            ),
        ))

    # Text traces (initial R² per model)
    for i, model in enumerate(models):
        initial_traces.append(go.Scatter(
            x=[txt_x],
            y=[txt_y],
            mode="text",
            text=[_r2_text(initial_age, model)],
            textfont=dict(size=18, color="#1A1A1A", family="Georgia, serif"),
            showlegend=False,
            hoverinfo="skip",
            visible=(i == 0),
        ))

    fig = go.Figure(data=initial_traces, frames=frames)

    # 1:1 reference line (static)
    fig.add_shape(
        type="line",
        x0=vmin, y0=vmin, x1=vmax, y1=vmax,
        line=dict(color=_COLORS["rust"], dash="dash", width=1.8),
        layer="below",
    )
    fig.add_annotation(
        x=vmax * 0.93, y=vmax * 0.86,
        text="1:1 Line",
        showarrow=False,
        font=dict(color=_COLORS["rust"], size=10, family="Georgia, serif"),
    )

    # ---- Slider steps ----------------------------------------------------
    slider_steps = [
        dict(
            args=[[str(age)], dict(
                frame=dict(duration=200, redraw=False),
                mode="immediate",
                transition=dict(duration=150),
            )],
            label=str(age),
            method="animate",
        )
        for age in ages
    ]

    fig.update_layout(
        **_LAYOUT_BASE,
        title="Stand Volume Predictions Across Tree Age",
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
        showlegend=False,
        # ---- Model tab buttons + Play / Pause ----------------------------
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                active=0,
                showactive=True,
                buttons=[
                    dict(
                        label=model,
                        method="update",
                        args=[
                            {"visible": [j == i for j in range(n_models)] + [j == i for j in range(n_models)]},
                        ],
                    )
                    for i, model in enumerate(models)
                ],
                x=0.0, xanchor="left",
                y=1.12, yanchor="top",
                bgcolor="#F5F0E8",
                bordercolor=_COLORS["sage"],
                font=dict(size=13, color="#1A1A1A", family="Georgia, serif"),
            ),
            dict(
                type="buttons",
                showactive=False,
                y=1.12,
                x=1.0,
                xanchor="right",
                direction="left",
                bgcolor="#F5F0E8",
                bordercolor=_COLORS["sage"],
                font=dict(color="#1A1A1A", size=11, family="Georgia, serif"),
                buttons=[
                    dict(
                        label="\u25b6  Play",
                        method="animate",
                        args=[None, dict(
                            frame=dict(duration=400, redraw=False),
                            fromcurrent=True,
                            transition=dict(duration=200, easing="quadratic-in-out"),
                        )],
                    ),
                    dict(
                        label="\u23f8  Pause",
                        method="animate",
                        args=[[None], dict(
                            frame=dict(duration=0, redraw=False),
                            mode="immediate",
                            transition=dict(duration=0),
                        )],
                    ),
                ],
            ),
        ],
        # ---- Age slider --------------------------------------------------
        sliders=[dict(
            active=0,
            steps=slider_steps,
            currentvalue=dict(
                prefix="Stand Age: ",
                suffix=" years",
                font=dict(size=13, color="#1A1A1A", family="Georgia, serif"),
                visible=True,
            ),
            pad=dict(t=55, b=10),
            tickcolor=_COLORS["sage"],
            bgcolor="#F5F0E8",
            bordercolor=_COLORS["sage"],
            font=dict(color="#1A1A1A", size=14),
            transition=dict(duration=300, easing="cubic-in-out"),
        )],
    )
    return fig


def plot_p2_scatter_tph(df: pd.DataFrame) -> go.Figure:
    """
    Scatter subplots of observed vs. predicted stand volume split by TPH
    (618 / 1236 / 1853 trees per hectare).  Points are coloured by stand age.
    Tab buttons switch between models; each subplot shows its own R² value.

    Parameters
    ----------
    df : pd.DataFrame  (proj2.csv)

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if df.empty:
        return _empty_figure("No data available — place proj2.csv in data/.")

    df = df.copy().dropna(subset=["volume_m", "predicted_volume_m", "tph", "age"])

    # Fixed TPH order, keep only values present in the data
    tph_vals = [t for t in [618, 1236, 1853] if t in df["tph"].values]
    n_tph = len(tph_vals)

    models = sorted(df["model"].dropna().unique().tolist())
    n_models = len(models)

    ages = sorted(df["age"].dropna().unique().tolist())
    n_ages = len(ages)
    age_colors = {age: _COLOR_SEQ[i % len(_COLOR_SEQ)] for i, age in enumerate(ages)}

    # Stable axis range across all subplots
    all_vals = pd.concat([df["volume_m"], df["predicted_volume_m"]]).dropna()
    vmin, vmax = all_vals.min() * 0.97, all_vals.max() * 1.03

    # Pre-compute R² per (model, tph)
    r2_map: dict = {}
    for model in models:
        for tph in tph_vals:
            sub = df[(df["model"] == model) & (df["tph"] == tph)]
            if len(sub) > 1:
                ss_res = ((sub["predicted_volume_m"] - sub["volume_m"]) ** 2).sum()
                ss_tot = ((sub["volume_m"] - sub["volume_m"].mean()) ** 2).sum()
                r2_map[(model, tph)] = (1 - ss_res / ss_tot) if ss_tot != 0 else 0.0
            else:
                r2_map[(model, tph)] = float("nan")

    subplot_titles = [f"{t:,} TPH" for t in tph_vals]
    fig = make_subplots(
        rows=n_tph, cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.12,
    )

    # Capture subplot-title annotations before we add our own
    # Reduce font so titles don't overlap on narrow/mobile screens
    subplot_title_anns = list(fig.layout.annotations)
    for ann in subplot_title_anns:
        ann["font"] = dict(size=10, family="Georgia, serif", color="#1A1A1A")

    # ---- Traces: model × tph × age — sampled to keep payload small --------
    for i, model in enumerate(models):
        model_df = df[df["model"] == model]
        for j, tph_val in enumerate(tph_vals):
            tph_df = model_df[model_df["tph"] == tph_val]
            for k, age_val in enumerate(ages):
                sub = _sample(tph_df[tph_df["age"] == age_val], _TPH_SAMPLE)
                fig.add_trace(
                    go.Scatter(
                        x=sub["volume_m"],
                        y=sub["predicted_volume_m"],
                        mode="markers",
                        name=f"Age {int(age_val)}",
                        legendgroup=str(int(age_val)),
                        showlegend=(j == 0),        # one legend entry per age
                        marker=dict(
                            color=age_colors[age_val],
                            size=6,
                            opacity=0.72,
                            line=dict(width=0.5, color="#FFFFFF"),
                        ),
                        hovertemplate=(
                            f"<b>Age:</b> {int(age_val)} yrs<br>"
                            "<b>Observed Vol.:</b> %{x:.3f} m\u00b3<br>"
                            "<b>Predicted Vol.:</b> %{y:.3f} m\u00b3<br>"
                            f"<b>TPH:</b> {tph_val}<br>"
                            f"<b>Model:</b> {model}"
                            "<extra></extra>"
                        ),
                        visible=(i == 0),
                    ),
                    row=j + 1, col=1,
                )

    # ---- 1:1 reference lines -----------------------------------------------
    for j in range(n_tph):
        xref = "x" if j == 0 else f"x{j + 1}"
        yref = "y" if j == 0 else f"y{j + 1}"
        fig.add_shape(
            type="line",
            x0=vmin, y0=vmin, x1=vmax, y1=vmax,
            line=dict(color=_COLORS["rust"], dash="dash", width=1.5),
            xref=xref, yref=yref,
        )

    # ---- R² annotation builder (one box per subplot) -----------------------
    def _r2_anns(model: str) -> list:
        anns = []
        for j, tph_val in enumerate(tph_vals):
            r2 = r2_map.get((model, tph_val), float("nan"))
            r2_text = f"R\u00b2 = {r2:.2f}" if not pd.isna(r2) else "R\u00b2 = N/A"
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

    # ---- Tab buttons -------------------------------------------------------
    buttons = []
    for i_sel, model in enumerate(models):
        vis = [i == i_sel for i in range(n_models) for _ in range(n_tph * n_ages)]
        buttons.append(dict(
            label=model,
            method="update",
            args=[
                {"visible": vis},
                {
                    "title": f"Stand Volume by Stand Density \u2014 {model}",
                    "annotations": subplot_title_anns + _r2_anns(model),
                },
            ],
        ))

    # ---- Layout ------------------------------------------------------------
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
        **_LAYOUT_BASE,
        title=f"Stand Volume by Stand Density \u2014 {models[0]}",
        annotations=subplot_title_anns + _r2_anns(models[0]),
        showlegend=True,
        legend=dict(
            orientation="h",
            title=dict(text="Stand Age (yrs):", font=dict(color="#1A1A1A")),
            bgcolor="#F5F0E8",
            bordercolor=_COLORS["sage"],
            borderwidth=1,
            font=dict(color="#1A1A1A"),
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
        ),
        updatemenus=[dict(
            type="buttons",
            direction="right",
            active=0,
            showactive=True,
            buttons=buttons,
            x=0.0, xanchor="left",
            y=1.16, yanchor="top",
            bgcolor="#F5F0E8",
            bordercolor=_COLORS["sage"],
            font=dict(size=14, color="#1A1A1A", family="Georgia, serif"),
        )],
        # Row 1 (top): y-title; x-title only if it's also the only row
        xaxis=dict(ax_style, title=dict(
            text=x_title if n_tph == 1 else "", font=_tfont)),
        yaxis=dict(ax_style, title=dict(text=y_title, font=_tfont)),
    )
    for j in range(1, n_tph):
        is_last = (j == n_tph - 1)
        fig.update_layout(**{
            f"xaxis{j + 1}": dict(ax_style, title=dict(
                text=x_title if is_last else "", font=_tfont)),
            f"yaxis{j + 1}": dict(ax_style, title=dict(text="", font=_tfont)),
        })
    fig.update_layout(height=780, margin=dict(l=70, r=30, t=110, b=160))
    return fig


def plot_p2_temporal_importance(df: pd.DataFrame) -> go.Figure:
    """
    Line chart showing how feature importance (% increase in MSE) shifts
    across age groups for each predictor variable.

    Shows top-10 features per model (by average importance across age groups).
    x-axis   : age_group          — categorical stand-age band
    y-axis   : pct_increase_in_mse — permutation importance metric
    lines    : feature             — one coloured line per predictor
    selector : model               — tab buttons to switch between models

    Parameters
    ----------
    df : pd.DataFrame  (proj2_age_importance.csv)

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if df.empty:
        return _empty_figure("No data available — place proj2_age_importance.csv in data/.")

    df = df.copy().dropna(subset=["age_group", "feature", "pct_increase_in_mse"])

    # Sort age groups by first numeric token so they plot chronologically
    try:
        df["_sort_key"] = (
            df["age_group"]
            .str.extract(r"(\d+)")[0]
            .astype(float)
        )
        df = df.sort_values("_sort_key")
    except Exception:
        df = df.sort_values("age_group")

    models = df["model"].dropna().unique().tolist()

    # Build all traces; track visibility per model
    all_traces: list[go.BaseTraceType] = []
    visibility_map: dict[str, list[bool]] = {m: [] for m in models}

    for model in models:
        model_df = df[df["model"] == model]

        # Top-10 features by average importance across age groups
        feat_avg = model_df.groupby("feature")["pct_increase_in_mse"].mean()
        top_features = feat_avg.nlargest(10).index.tolist()
        # Sort by descending average importance for consistent colour assignment
        top_features = sorted(top_features, key=lambda f: feat_avg[f], reverse=True)

        for f_idx, feature in enumerate(top_features):
            feat_df = model_df[model_df["feature"] == feature].sort_values(
                "_sort_key" if "_sort_key" in model_df.columns else "age_group"
            )
            legend_label = _FEATURE_LABELS.get(feature, feature)

            trace = go.Scatter(
                x=feat_df["age_group"],
                y=feat_df["pct_increase_in_mse"],
                mode="lines+markers",
                name=legend_label,
                line=dict(
                    color=_COLOR_SEQ[f_idx % len(_COLOR_SEQ)],
                    width=2.4,
                ),
                marker=dict(size=7, symbol="circle"),
                hovertemplate=(
                    f"<b>Feature:</b> {feature}<br>"
                    "<b>Age Group:</b> %{x}<br>"
                    "<b>% Inc. MSE:</b> %{y:.2f}%"
                    f"<br><b>Model:</b> {model}"
                    "<extra></extra>"
                ),
                visible=(model == models[0]),
            )
            all_traces.append(trace)

            for m in models:
                visibility_map[m].append(m == model)

    fig = go.Figure(data=all_traces)

    buttons = [
        dict(
            label=model,
            method="update",
            args=[
                {"visible": visibility_map[model]},
                {"title": f"Feature Importance Over Stand Age \u2014 {model}"},
            ],
        )
        for model in models
    ]

    fig.update_layout(
        **_LAYOUT_BASE,
        height=700,
        title=f"Feature Importance Over Stand Age \u2014 {models[0]}",
        xaxis=dict(
            title="Stand Age (years)",
            gridcolor="#EAE6DE",
            tickangle=0,
            title_font=dict(color="#1A1A1A"),
            tickfont=dict(color="#1A1A1A"),
        ),
        yaxis=dict(
            title="% Increase in MSE (Importance)",
            gridcolor="#EAE6DE",
            zeroline=True,
            zerolinecolor="#D8D3CB",
            title_font=dict(color="#1A1A1A"),
            tickfont=dict(color="#1A1A1A"),
        ),
        legend=dict(
            title="Feature",
            bgcolor="#F5F0E8",
            bordercolor=_COLORS["sage"],
            borderwidth=1,
            font=dict(color="#1A1A1A", size=11),
            title_font=dict(color="#1A1A1A"),
            entrywidth=0,
            yanchor="top",
            y=-0.18,
            xanchor="left",
            x=0,
        ),
        updatemenus=[dict(
            type="buttons",
            direction="right",
            active=0,
            showactive=True,
            buttons=buttons,
            x=0.0, xanchor="left",
            y=1.12, yanchor="top",
            bgcolor="#F5F0E8",
            bordercolor=_COLORS["sage"],
            font=dict(size=13, color="#1A1A1A", family="Georgia, serif"),
        )],
    )
    # Extra bottom margin so the below-chart legend isn't clipped
    fig.update_layout(margin=dict(l=70, r=50, t=130, b=220))
    return fig


# ---------------------------------------------------------------------------
# Growth-curve plot
# ---------------------------------------------------------------------------

def plot_p2_growth_curve(df: pd.DataFrame) -> go.Figure:
    """
    Growth curve of mean observed vs. mean predicted volume across stand age,
    with 25th–75th percentile ribbon bands.  Tab buttons switch between models.

    x-axis  : age                — stand age (years, integer ticks)
    y-axis  : volume_m           — m³ tree⁻¹
    ribbons : Q25–Q75 of field and predicted volumes
    selector: model              — RF / SVR tab buttons

    Parameters
    ----------
    df : pd.DataFrame  (proj2.csv)

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if df.empty:
        return _empty_figure("No data available — place proj2.csv in data/.")

    df = df.copy().dropna(subset=["volume_m", "predicted_volume_m", "age", "model"])

    models = sorted(df["model"].dropna().unique().tolist())

    # Colours — matching figure style: indigo for field, green for predicted
    _FIELD_COLOR = "#3e4989"
    _FIELD_FILL  = "rgba(62,73,137,0.25)"
    _PRED_COLOR  = "#35b779"
    _PRED_FILL   = "rgba(53,183,121,0.22)"

    # Stable y-axis range across all models
    all_vol = pd.concat([df["volume_m"], df["predicted_volume_m"]]).dropna()
    vmin = all_vol.min() * 0.88
    vmax = 0.7

    TRACES_PER_MODEL = 4
    all_traces: list[go.BaseTraceType] = []

    for i_m, model in enumerate(models):
        mdf = df[df["model"] == model]
        agg = (
            mdf.groupby("age")
            .agg(
                field_mean=("volume_m",           "mean"),
                field_q25= ("volume_m",           lambda x: x.quantile(0.25)),
                field_q75= ("volume_m",           lambda x: x.quantile(0.75)),
                pred_mean= ("predicted_volume_m", "mean"),
                pred_q25=  ("predicted_volume_m", lambda x: x.quantile(0.25)),
                pred_q75=  ("predicted_volume_m", lambda x: x.quantile(0.75)),
            )
            .reset_index()
            .sort_values("age")
        )

        ages_fwd = agg["age"].tolist()
        ages_rev = list(reversed(ages_fwd))
        visible  = (i_m == 0)

        # -- Ribbon: field (closed polygon) ----------------------------------
        x_field = ages_fwd + ages_rev
        y_field = agg["field_q75"].tolist() + list(reversed(agg["field_q25"].tolist()))
        all_traces.append(go.Scatter(
            x=x_field, y=y_field,
            fill="toself",
            fillcolor=_FIELD_FILL,
            line=dict(color="rgba(0,0,0,0)", width=0),
            name="25th–75th Percentile (Field)",
            showlegend=True,
            hoverinfo="skip",
            visible=visible,
        ))

        # -- Ribbon: predicted (closed polygon) ------------------------------
        x_pred = ages_fwd + ages_rev
        y_pred = agg["pred_q75"].tolist() + list(reversed(agg["pred_q25"].tolist()))
        all_traces.append(go.Scatter(
            x=x_pred, y=y_pred,
            fill="toself",
            fillcolor=_PRED_FILL,
            line=dict(color="rgba(0,0,0,0)", width=0),
            name="25th–75th Percentile (Predicted)",
            showlegend=True,
            hoverinfo="skip",
            visible=visible,
        ))

        # -- Mean field volume (dashed + circle markers) ---------------------
        all_traces.append(go.Scatter(
            x=ages_fwd,
            y=agg["field_mean"].tolist(),
            mode="lines+markers",
            name="Mean Field Volume",
            line=dict(color=_FIELD_COLOR, dash="dash", width=2.4),
            marker=dict(symbol="circle", size=9, color=_FIELD_COLOR),
            hovertemplate=(
                "<b>Age:</b> %{x} yrs<br>"
                "<b>Mean Field Vol.:</b> %{y:.3f} m\u00b3 tree\u207b\u00b9<br>"
                f"<b>Model:</b> {model}"
                "<extra></extra>"
            ),
            visible=visible,
        ))

        # -- Mean predicted volume (solid line) ------------------------------
        all_traces.append(go.Scatter(
            x=ages_fwd,
            y=agg["pred_mean"].tolist(),
            mode="lines+markers",
            name="Mean Predicted Volume",
            line=dict(color=_PRED_COLOR, width=2.4),
            marker=dict(symbol="circle", size=7, color=_PRED_COLOR),
            hovertemplate=(
                "<b>Age:</b> %{x} yrs<br>"
                "<b>Mean Predicted Vol.:</b> %{y:.3f} m\u00b3 tree\u207b\u00b9<br>"
                f"<b>Model:</b> {model}"
                "<extra></extra>"
            ),
            visible=visible,
        ))

    # -- Tab buttons ---------------------------------------------------------
    n_models = len(models)
    buttons = []
    for i_sel, model in enumerate(models):
        vis = [
            (i_m == i_sel)
            for i_m in range(n_models)
            for _ in range(TRACES_PER_MODEL)
        ]
        buttons.append(dict(
            label=model,
            method="update",
            args=[
                {"visible": vis},
                {"title": f"Volume Growth Curves \u2014 {model}"},
            ],
        ))

    fig = go.Figure(data=all_traces)

    fig.update_layout(
        **_LAYOUT_BASE,
        title=f"Volume Growth Curves \u2014 {models[0]}",
        xaxis=dict(
            title="Age (years)",
            tickmode="linear",
            dtick=1,
            gridcolor="#EAE6DE",
            zeroline=False,
            title_font=dict(color="#1A1A1A"),
            tickfont=dict(color="#1A1A1A"),
        ),
        yaxis=dict(
            title="Volume (m\u00b3 tree\u207b\u00b9)",
            range=[vmin, vmax],
            gridcolor="#EAE6DE",
            zeroline=False,
            title_font=dict(color="#1A1A1A"),
            tickfont=dict(color="#1A1A1A"),
        ),
        showlegend=True,
        legend=dict(
            bgcolor="#F5F0E8",
            bordercolor=_COLORS["sage"],
            borderwidth=1,
            font=dict(color="#1A1A1A", size=12, family="Georgia, serif"),
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.01,
        ),
        updatemenus=[dict(
            type="buttons",
            direction="right",
            active=0,
            showactive=True,
            buttons=buttons,
            x=0.0, xanchor="left",
            y=1.12, yanchor="top",
            bgcolor="#F5F0E8",
            bordercolor=_COLORS["sage"],
            font=dict(size=13, color="#1A1A1A", family="Georgia, serif"),
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
