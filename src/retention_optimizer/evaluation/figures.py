"""Reusable figure builders for the budget sweeps.

Pure plotting: every function takes an already-computed sweep DataFrame and
returns a matplotlib figure. Nothing here recomputes a sweep, and nothing here
hardcodes a margin factor — it arrives as an argument and reaches the title on
its own, so the dashboard can redraw any of these live as the user moves the
slider.
"""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import StrMethodFormatter

# Per-strategy colours, so the same strategy keeps its colour across figures.
DEFAULT_COLORS = {
    "random": "#898781",
    "by_churn": "#2a78d6",
    "by_priority": "#eb6834",
    "optimal": "#1baf7a",
}
# Used for strategy keys the palette does not know about (e.g. a fifth strategy
# added in S5), so an unknown name draws in a distinct colour instead of failing.
FALLBACK_COLORS = ["#4a3aa7", "#c9a227", "#b5446e", "#2f8f83"]

INK, INK_SOFT, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID_COLOR, AXIS_COLOR = "#e1e0d9", "#c3c2b7"

# Marker-label layout, in typographic points. The horizontal offset has to clear
# the marker itself (radius ~5pt plus a white edge); the vertical bias keeps the
# middle label of an odd-sized group off its own dot.
LABEL_X_OFFSET = 16
LABEL_STEP = 15
LABEL_BIAS = 5
# Fraction of the y-range kept free at each end so a stacked label never lands
# outside the plotting area.
LABEL_MARGIN_FRAC = 0.03


def _color_for(key: str, position: int) -> str:
    """Colour for a strategy key, falling back for unknown names."""
    return DEFAULT_COLORS.get(key, FALLBACK_COLORS[position % len(FALLBACK_COLORS)])


def _style_axes(ax: plt.Axes, ylabel: str, yfmt: str) -> None:
    """Thousands separators, recessive grid and axes, no top/right spines."""
    ax.set_xlabel("Retention budget (EUR)")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.yaxis.set_major_formatter(StrMethodFormatter(yfmt))
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS_COLOR)
    ax.tick_params(colors=MUTED, labelsize=9)


def _axes_height_points(ax: plt.Axes) -> float:
    """Height of the axes in typographic points, to convert offsets to data units."""
    return ax.figure.get_size_inches()[1] * 72.0 * ax.get_position().height


def _annotate_group(
    ax: plt.Axes,
    rows: list,
    ycol: str,
    colors: dict,
    annotate: callable,
    budget: float,
) -> None:
    """Label every strategy at one budget, stacked around the group's centre.

    Spreading up *and* down from the median keeps the block of labels sitting on
    its own cloud of points; stacking only downwards, as an earlier version did,
    pushed the bottom label off the axes once four lines bunched together.
    """
    if not rows:
        return

    span = (len(rows) - 1) / 2
    ymin, ymax = ax.get_ylim()
    per_point = (ymax - ymin) / _axes_height_points(ax)
    margin = (ymax - ymin) * LABEL_MARGIN_FRAC

    for rank, row in enumerate(rows):
        dy = LABEL_BIAS + (span - rank) * LABEL_STEP

        # Pull back inside the axes if the stack would overflow either end.
        y_label = row[ycol] + dy * per_point
        if y_label > ymax - margin:
            dy = (ymax - margin - row[ycol]) / per_point
        elif y_label < ymin + margin:
            dy = (ymin + margin - row[ycol]) / per_point

        ax.annotate(
            annotate(row[ycol]),
            xy=(budget, row[ycol]),
            xytext=(LABEL_X_OFFSET, dy),
            textcoords="offset points",
            fontsize=9,
            color=colors[row.strategy],
            fontweight="bold",
            zorder=5,
            # On steep stretches the curve keeps climbing into wherever the
            # label lands, so no offset alone keeps text off the lines. A
            # near-opaque plate does, and stays geometry-independent.
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
            },
        )


def _draw_strategies(
    ax: plt.Axes,
    df_sweep: pd.DataFrame,
    ycol: str,
    strategy_labels: dict,
    budget_markers: tuple,
    annotate: callable,
) -> None:
    """Draw one line per strategy, with a highlighted marker at each budget."""
    colors = {
        key: _color_for(key, position) for position, key in enumerate(strategy_labels)
    }

    for key, label in strategy_labels.items():
        d = df_sweep[df_sweep.strategy == key]
        ax.plot(
            d.budget, d[ycol], color=colors[key], linewidth=2.2, label=label, zorder=3
        )

        marks = d[d.budget.isin(budget_markers)]
        ax.scatter(
            marks.budget,
            marks[ycol],
            s=70,
            color=colors[key],
            zorder=4,
            edgecolor="white",
            linewidth=1.2,
        )

    for budget in budget_markers:
        ax.axvline(budget, color=AXIS_COLOR, linewidth=0.8, linestyle=":", zorder=1)

    # Labels are laid out per budget rather than per strategy: at a given budget
    # the lines can sit arbitrarily close, so a fixed per-strategy offset would
    # collide as soon as the margin factor changed the spacing. Done after the
    # lines so the y-limits are settled and the clamp below can use them.
    for budget in budget_markers:
        rows = [
            row
            for _, row in df_sweep[df_sweep.budget == budget]
            .sort_values(ycol, ascending=False)
            .iterrows()
            if row.strategy in strategy_labels
        ]
        _annotate_group(ax, rows, ycol, colors, annotate, budget)


def plot_value_curve(
    df_sweep: pd.DataFrame,
    margin_factor: float,
    strategy_labels: dict,
    budget_markers: tuple = (5000, 10000, 20000),
    title: str | None = None,
    savepath: str | None = None,
) -> Figure:
    """Expected value retained against budget, one line per strategy.

    Warning:
        `df_sweep` must have been produced with the *same* `margin_factor` that
        is passed here. The margin factor scales every euro in the sweep, but
        this function only plots and labels — it cannot detect a mismatch, and a
        wrong one would draw correct numbers under a wrong caption. Keeping
        `sweep_strategies(..., margin_factor=X)` and
        `plot_value_curve(..., margin_factor=X)` in step is the caller's job.

    Args:
        df_sweep: Long-format sweep with `budget`, `strategy`, `expected_value`.
        margin_factor: The margin factor the sweep was computed with; goes into
            the title.
        strategy_labels: Mapping of strategy key to business-readable label.
            Also fixes the plotting order.
        budget_markers: Budgets to call out with a dot and a value annotation.
        title: Overrides the default title; the margin factor is appended either
            way.
        savepath: If given, the figure is written there at 150 dpi.

    Returns:
        The matplotlib figure, so the caller can adjust it further.
    """
    fig, ax = plt.subplots(figsize=(10, 6.5))

    _draw_strategies(
        ax,
        df_sweep,
        "expected_value",
        strategy_labels,
        budget_markers,
        lambda v: f"{v / 1000:,.1f}k",
    )

    base = title or "How much value does each way of spending the budget retain?"
    _style_axes(ax, "Expected value retained (EUR)", "{x:,.0f}")
    ax.set_title(
        f"{base}  (margin factor = {margin_factor})",
        fontsize=13,
        loc="left",
        pad=14,
        color=INK,
    )
    ax.legend(loc="upper left", frameon=False, fontsize=10.5)

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig


def plot_efficiency_curve(
    df_sweep: pd.DataFrame,
    margin_factor: float,
    strategy_labels: dict,
    budget_markers: tuple = (5000, 10000, 20000),
    title: str | None = None,
    savepath: str | None = None,
) -> Figure:
    """Euros retained per euro spent, one line per strategy.

    Draws the break-even line at 1.0: below it a strategy returns less than it
    costs, which the absolute-value chart cannot show.

    Warning:
        Same margin-factor contract as `plot_value_curve` — the sweep must have
        been computed with the value passed here. Note that efficiency is
        *linear* in the margin factor (it divides two quantities, only one of
        which scales), so unlike the strategy ranking it is not invariant to it:
        whether a line sits above or below break-even depends on the margin
        factor being right.

    Args:
        df_sweep: Long-format sweep with `budget`, `strategy`,
            `expected_value`, `spend`.
        margin_factor: The margin factor the sweep was computed with.
        strategy_labels: Mapping of strategy key to business-readable label.
        budget_markers: Budgets to call out with a dot and a value annotation.
        title: Overrides the default title; the margin factor is appended.
        savepath: If given, the figure is written there at 150 dpi.

    Returns:
        The matplotlib figure.
    """
    d = df_sweep.copy()
    d["efficiency"] = d.expected_value / d.spend

    fig, ax = plt.subplots(figsize=(10, 6.5))

    _draw_strategies(
        ax,
        d,
        "efficiency",
        strategy_labels,
        budget_markers,
        lambda v: f"{v:.2f}x",
    )

    ax.axhline(1.0, color=INK_SOFT, linewidth=1.2, linestyle="--", zorder=2)
    ax.annotate(
        "break-even (EUR retained = EUR spent)",
        xy=(0.99, 1.0),
        xycoords=("axes fraction", "data"),
        xytext=(0, 5),
        textcoords="offset points",
        fontsize=9,
        color=INK_SOFT,
        ha="right",
    )

    base = title or "Efficiency: euros retained per euro spent"
    _style_axes(ax, "Euros retained per euro spent", "{x:,.1f}x")
    ax.set_title(
        f"{base}  (margin factor = {margin_factor})",
        fontsize=13,
        loc="left",
        pad=14,
        color=INK,
    )
    ax.legend(loc="upper right", frameon=False, fontsize=10.5)

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig
