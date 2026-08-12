"""Plotly (dark) charts for the dashboard.

Plotting only: these functions never recompute a sweep. Whatever valuation
parameters produced `df_sweep` are the caller's responsibility; the margin
factor is passed in purely so the title can state it.
"""

import pandas as pd
import plotly.graph_objects as go

from app.theme import BG, NEUTRAL, TEXT


def plot_value_curve_plotly(
    df_sweep: pd.DataFrame,
    strategy_labels: dict,
    strategy_colors: dict,
    current_budget: float,
    margin_factor: float,
) -> go.Figure:
    """Net value retained against budget, one line per strategy.

    "Net" means expected revenue retained minus the contact cost actually spent,
    so the curve answers what the company keeps rather than what it grosses. The
    column is derived here from `expected_value` and `spend`.

    Args:
        df_sweep: Long-format sweep with `budget`, `strategy`, `expected_value`
            and `spend`.
        strategy_labels: Strategy key to business-readable name. Also fixes the
            drawing order and which strategies appear at all.
        strategy_colors: Strategy key to colour.
        current_budget: Budget the slider sits at; drawn as a vertical marker.
        margin_factor: The margin factor the sweep was computed with. Only
            labels the chart — it cannot be verified from the data.

    Returns:
        A dark-themed Plotly figure.
    """
    d = df_sweep.copy()
    d["net_value"] = d.expected_value - d.spend

    fig = go.Figure()

    for key, label in strategy_labels.items():
        series = d[d.strategy == key]
        if series.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=series.budget,
                y=series.net_value,
                name=label,
                mode="lines",
                line={"color": strategy_colors[key], "width": 3.5, "shape": "spline"},
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    "Budget: €%{x:,.0f}<br>"
                    "Net value retained: €%{y:,.0f}<extra></extra>"
                ),
            )
        )

    fig.add_vline(
        x=current_budget,
        line={"color": NEUTRAL, "width": 1.5, "dash": "dot"},
        annotation_text=f"€{current_budget:,.0f}",
        annotation_position="top",
        annotation_font={"color": NEUTRAL, "size": 11},
    )

    fig.update_layout(
        title={
            "text": (
                "Net value retained by budget"
                f"<br><span style='font-size:12px;color:{NEUTRAL}'>"
                f"margin factor = {margin_factor} "
                "(EUR of margin per EUR of revenue)</span>"
            ),
            "font": {"size": 17, "color": TEXT},
            "y": 0.97,
            "yanchor": "top",
        },
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font={"color": TEXT, "size": 12},
        hovermode="x unified",
        # Legend sits below the two-line title block rather than at the usual
        # y=1.02, which would land on the margin-factor subtitle.
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "x": 0,
            "bgcolor": "rgba(0,0,0,0)",
        },
        margin={"l": 70, "r": 30, "t": 125, "b": 60},
        height=460,
    )

    axis_style = {
        "gridcolor": "#232838",
        "zerolinecolor": "#2E3444",
        "linecolor": "#2E3444",
        "tickfont": {"color": NEUTRAL},
        "title_font": {"color": TEXT, "size": 12},
    }
    fig.update_xaxes(
        title_text="Retention budget (EUR)", tickformat=",.0f", **axis_style
    )
    fig.update_yaxes(
        title_text=("Net value retained (EUR) · expected revenue − contact cost"),
        tickformat=",.0f",
        **axis_style,
    )

    return fig
