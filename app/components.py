"""The decision view: KPIs, curve and recommended-customer list.

`render_decision_view` is parameterised by cost regime rather than written for a
particular screen. The flat tab instantiates it with one contact cost for
everyone and ranking by value as the recommendation; the channel tab passes a
per-customer cost vector and the knapsack. Same layout, same units, same
guarantees — only the cost and the recommended strategy differ.

Sliders live outside this function, in `sidebar_controls()`: two tabs render the
view on the same page, and Streamlit widgets cannot be created twice with the
same identity.
"""

import numpy as np
import pandas as pd
import streamlit as st

from app.plotting import plot_value_curve_plotly
from app.theme import (
    ACCENT,
    NEUTRAL,
    SIMULATED,
    STRATEGY_COLORS,
    STRATEGY_LABELS,
    TEXT,
)
from app.uplift import (
    CONTRACT_SEGMENTS,
    UNIFORM_EFFECTIVENESS,
    effectiveness_from_contract,
)
from retention_optimizer.evaluation.strategies import (
    channel_cost_from_contract,
    strategy_by_churn,
    strategy_by_priority,
    strategy_optimal,
)
from retention_optimizer.evaluation.sweep import sweep_strategies
from retention_optimizer.optimization.value import annuity_factor

# Coarser than the notebook sweep (step 1000): the curve is redrawn whenever a
# slider moves. The channel grid is coarser still because every point there is a
# MILP solve rather than a sort.
BUDGET_GRID = {
    "flat": np.arange(1000, 80001, 2000),
    "channel": np.arange(1000, 80001, 3000),
}

# With a flat cost the ranking by p*m is already optimal, so the knapsack would
# draw on top of it and add nothing; once costs differ it is the recommendation.
RECOMMENDED_STRATEGY = {"flat": "by_priority", "channel": "optimal"}

# What the headline advantage is measured against. Each tab answers a different
# question: is weighting by value worth it, and is optimising worth it on top.
ADVANTAGE_BASELINE = {"flat": "by_churn", "channel": "by_priority"}
ADVANTAGE_TITLE = {
    "flat": "Advantage vs risk targeting",
    "channel": "Advantage of optimising vs prioritising",
}

# Strategies drawn on each tab, in drawing order.
CURVE_STRATEGIES = {
    "flat": ("by_churn", "by_priority"),
    "channel": ("by_churn", "by_priority", "optimal"),
}
# On the channel tab the two teal lines need a second cue beyond brightness.
CURVE_DASHES = {"flat": {}, "channel": {"by_priority": "dash"}}

STRATEGY_FUNCS = {
    "by_churn": strategy_by_churn,
    "by_priority": strategy_by_priority,
    "optimal": strategy_optimal,
}


def sidebar_controls() -> dict:
    """Scenario sliders, shared by every tab. Each label carries its unit."""
    st.sidebar.header("Scenario")

    budget = st.sidebar.slider(
        "Budget (EUR)",
        min_value=1000,
        max_value=80000,
        value=10000,
        step=1000,
        help="Total money available for the retention campaign.",
    )
    cost = st.sidebar.slider(
        "Cost per contact (EUR) — flat-cost tab only",
        min_value=5,
        max_value=50,
        value=20,
        step=1,
        help=(
            "What one retention action costs. Applies to the flat-cost tab. "
            "The channel tab ignores it and prices each customer by contract "
            "type instead."
        ),
    )
    effectiveness = st.sidebar.slider(
        "Effectiveness (share of targeted churners retained)",
        min_value=0.05,
        max_value=0.50,
        value=0.30,
        step=0.05,
        help="Of the customers who would have left, the fraction the action saves.",
    )
    margin_factor = st.sidebar.slider(
        "Margin factor (EUR margin / EUR revenue · 0.35 ≈ telco EBITDA)",
        min_value=0.10,
        max_value=0.45,
        value=0.35,
        step=0.05,
        help="How much of the billed revenue is actually margin.",
    )
    horizon = st.sidebar.slider(
        "Horizon (months of margin from a retained customer)",
        min_value=6,
        max_value=36,
        value=12,
        step=6,
        help="How many months of margin a saved customer is credited with.",
    )

    st.sidebar.caption(
        "Every euro on both tabs is computed under these assumptions. "
        "The contact cost is the only one that differs between tabs."
    )

    return {
        "budget": float(budget),
        "cost": float(cost),
        "effectiveness": float(effectiveness),
        "margin_factor": float(margin_factor),
        "horizon": int(horizon),
    }


@st.cache_data(show_spinner=False)
def _budget_curve(
    _p: np.ndarray,
    _m: np.ndarray,
    _cost: np.ndarray,
    _effectiveness,
    cost_mode: str,
    cost_key: float,
    effectiveness_key: tuple,
    margin_factor: float,
    horizon: int,
) -> pd.DataFrame:
    """Cached budget sweep behind the chart.

    Leading-underscore arguments are excluded from the cache key: the customer
    arrays never change within a session, and the cost and effectiveness arrays
    are fully determined by the keys that follow them. `effectiveness_key` is a
    tuple rather than a summary statistic: two different per-segment hypotheses
    can share a mean, and collapsing them would serve one the other's curve.
    Moving a slider
    invalidates the entry; re-rendering with untouched sliders does not, which
    matters most on the channel tab where every point is a MILP solve.
    """
    strategies = {k: STRATEGY_FUNCS[k] for k in CURVE_STRATEGIES[cost_mode]}
    return sweep_strategies(
        _p,
        _m,
        BUDGET_GRID[cost_mode],
        strategies,
        cost=_cost,
        effectiveness=_effectiveness,
        margin_factor=margin_factor,
        H=horizon,
    )


def _kpi_row(
    recommended_info: dict,
    comparison_info: dict,
    recommended: str,
    comparison: str,
    title: str,
    simulated: bool = False,
) -> None:
    """Five metrics, each stating its unit.

    The headline advantage is shown in euros *and* as a share of the baseline,
    together on purpose: the euro gap widens as the budget grows, while the
    relative gap is largest when the budget is tightest. Either number on its
    own tells half the story and points at the opposite conclusion about where
    the method matters most.
    """
    net_value = recommended_info["expected_value"] - recommended_info["spend"]
    comparison_net = comparison_info["expected_value"] - comparison_info["spend"]
    roi = recommended_info["expected_value"] / recommended_info["spend"]

    advantage = net_value - comparison_net

    ours = STRATEGY_LABELS[recommended]
    theirs = STRATEGY_LABELS[comparison]

    if simulated:
        # Under a simulated per-segment effectiveness the percentage stops being
        # trustworthy: `p × m` no longer tracks net value, so the ranking
        # baseline picks badly, its net collapses towards zero and the ratio
        # explodes precisely where the baseline is worst. The euro difference is
        # a subtraction and does not have that failure mode, so it stands alone.
        pct_text = ""
        note = (
            "Shown in euros: the relative % becomes unstable when the assumed "
            "effectiveness makes the ranking baseline retain very little (small "
            "denominator). The euro figure is the robust measure here."
        )
    else:
        # A percentage off a zero base would be meaningless rather than merely
        # large; the baseline can retain nothing under harsh slider settings.
        advantage_pct = advantage / comparison_net * 100 if comparison_net > 0 else None
        pct_text = (
            f'<span style="font-size:1.35rem;font-weight:600">'
            f"({advantage_pct:.1f}%)</span>"
            if advantage_pct is not None
            else '<span style="font-size:1.35rem;font-weight:600">(n/a)</span>'
        )
        note = (
            f"% = (net «{ours}» − net «{theirs}») / net «{theirs}»"
            " · net = expected revenue − contact cost"
        )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Recommended customers", f"{recommended_info['n_selected']:,}")
    c1.caption("people to contact")

    c2.metric("Net value retained (EUR)", f"€{net_value:,.0f}")
    c2.caption("expected revenue − contact cost")

    c3.metric("Budget used (EUR)", f"€{recommended_info['spend']:,.0f}")
    c3.caption("of the budget above")

    c4.metric("Return (EUR retained / EUR spent)", f"{roi:.2f}×")
    c4.caption("gross retained ÷ spend")

    badge = (
        f'<span style="color:{SIMULATED};font-weight:700"> [SIMULATED]</span>'
        if simulated
        else ""
    )
    c5.markdown(
        f"""
        <div style="padding-top:2px">
          <div style="color:{NEUTRAL};font-size:0.8rem">{title}{badge}</div>
          <div style="color:{ACCENT};font-size:2.1rem;font-weight:700;
                      line-height:1.25">
            +€{advantage:,.0f} {pct_text}
          </div>
          <div style="color:{NEUTRAL};font-size:0.7rem;line-height:1.35;
                      padding-top:2px">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _customer_table(
    mask: np.ndarray,
    p: np.ndarray,
    m: np.ndarray,
    ids: pd.Series,
    cost: np.ndarray,
    scalar,
    show_cost: bool,
    effectiveness=None,
) -> pd.DataFrame:
    """Recommended customers, ranked by priority, with gross and net value.

    `show_cost` adds the per-customer contact cost. It is only worth a column
    when the cost actually varies: on the channel tab it is what explains why a
    cheap customer can outrank an expensive one of similar value.

    `scalar` may be a float or a per-customer array — under simulated uplift the
    valuation multiplier differs by segment, so it has to broadcast.
    `effectiveness`, when given, shows which assumption was applied to each row.
    """
    recoverable = p * m * scalar
    columns = {
        "customerID": np.asarray(ids)[mask],
        "Churn risk (%)": (p[mask] * 100).round(1),
        "Monthly value (EUR/month)": m[mask].round(2),
        "Priority (p × m)": (p[mask] * m[mask]).round(2),
    }
    if show_cost:
        columns["Contact cost (EUR)"] = cost[mask].round(2)
    if effectiveness is not None:
        columns["Effectiveness (segment, assumed)"] = np.round(
            np.broadcast_to(effectiveness, p.shape)[mask], 2
        )
    columns["Recoverable value (EUR) — gross"] = recoverable[mask].round(2)
    columns["Net value (EUR) — after contact cost"] = (recoverable - cost)[mask].round(
        2
    )

    table = pd.DataFrame(columns)
    # Under uniform effectiveness `p × m` is the ranking, so it sorts the list.
    # Under simulated uplift it is not: effectiveness varies by segment, so the
    # quantity the optimiser actually maximises is the net value. Sorting by it
    # keeps the top of the list responsive to the sliders instead of burying
    # every change 300 rows down.
    sort_by = (
        "Net value (EUR) — after contact cost"
        if effectiveness is not None
        else "Priority (p × m)"
    )
    return table.sort_values(sort_by, ascending=False).reset_index(drop=True)


def _uplift_controls(contract: pd.Series):
    """Simulated per-segment effectiveness. Channel tab only.

    Returns `(effectiveness, key, simulating)`. With the toggle off this is the
    plain uniform scalar and the sliders render greyed out but visible, so the
    capability is discoverable without being imposed.

    Every slider starts at the uniform value rather than at some suggestive
    default. Switching simulation on and touching nothing must reproduce the
    real-data run exactly — otherwise the app would be nudging the user toward a
    conclusion and calling it their own.
    """
    simulating = st.toggle(
        "Simulate heterogeneous effectiveness by segment  [SIMULATED]",
        value=False,
        key="uplift_toggle",
        help=(
            "Explore what happens if the campaign works differently on different "
            "contract types. These are assumptions, not measurements."
        ),
    )

    cols = st.columns(3)
    effs = {}
    for col, segment in zip(cols, CONTRACT_SEGMENTS):
        effs[segment] = col.slider(
            f"{segment} · effectiveness",
            min_value=0.0,
            max_value=0.6,
            value=UNIFORM_EFFECTIVENESS,
            step=0.05,
            disabled=not simulating,
            key=f"uplift_{segment}",
            help="Share of would-be churners in this segment the action saves.",
        )

    if not simulating:
        return UNIFORM_EFFECTIVENESS, ("uniform", UNIFORM_EFFECTIVENESS), False

    st.markdown(
        f"""
        <div style="border-left:3px solid {SIMULATED};background:rgba(251,191,36,0.08);
                    padding:10px 14px;margin:6px 0 2px 0;border-radius:4px">
          <span style="color:{SIMULATED};font-weight:700">SIMULATION MODE</span>
          <span style="color:{TEXT};font-size:0.86rem"> — per-segment
          effectiveness is an <b>assumption you are exploring</b>, not a measured
          value. Real effectiveness would require an experiment (an A/B test);
          nothing in this project estimates it. Every figure below is a
          sensitivity analysis, not a prediction.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    effectiveness = effectiveness_from_contract(contract, effs)
    key = tuple(effs[s] for s in CONTRACT_SEGMENTS)
    return effectiveness, key, True


def _channel_breakdown(cost_vector: np.ndarray, contract: pd.Series) -> str:
    """One-line summary of the channel price list actually in force."""
    parts = []
    for name in ("Month-to-month", "One year", "Two year"):
        unit = cost_vector[(contract == name).to_numpy()]
        if len(unit):
            parts.append(f"{name} €{unit[0]:,.0f}")
    return " · ".join(parts) + f" · mean €{cost_vector.mean():,.2f}"


def render_decision_view(
    cost_mode: str,
    p: np.ndarray,
    m: np.ndarray,
    contract: pd.Series,
    ids: pd.Series,
    params: dict,
) -> None:
    """Render the decision screen for one cost regime.

    Args:
        cost_mode: `"flat"` (one cost for everyone, from the sidebar slider) or
            `"channel"` (a per-customer cost vector derived from `contract`).
        p: Calibrated churn probability per customer.
        m: Monthly margin per customer.
        contract: The raw `Contract` column, aligned to `ids`. Unused under
            `"flat"`; the channel regime prices every customer from it.
        ids: Customer identifiers, aligned positionally with `p` and `m`.
        params: Slider values from `sidebar_controls()`.
    """
    if cost_mode not in RECOMMENDED_STRATEGY:
        raise ValueError(f"unknown cost_mode: {cost_mode!r}")

    recommended = RECOMMENDED_STRATEGY[cost_mode]
    comparison = ADVANTAGE_BASELINE[cost_mode]

    if cost_mode == "flat":
        cost_vector = np.full(len(p), params["cost"])
        cost_key = params["cost"]
        regime_caption = (
            f"Flat contact cost of €{params['cost']:,.0f} for every customer"
        )
    else:
        # The channel policy is declared over an observable column, so the cost
        # vector is reproducible from the data rather than set by a slider.
        cost_vector = channel_cost_from_contract(contract)
        cost_key = float(cost_vector.sum())
        regime_caption = (
            "Contact cost by channel — "
            + _channel_breakdown(cost_vector, contract)
            + ". The sidebar's flat cost does not apply here."
        )

    st.markdown(f"#### {STRATEGY_LABELS[recommended]} — recommended plan")
    st.caption(regime_caption)

    # Effectiveness is a scalar everywhere except under simulated uplift, where
    # it becomes a per-customer vector. The strategies, the sweep and the value
    # formula all broadcast, so this is the only place that has to know.
    if cost_mode == "channel":
        effectiveness, eff_key, simulating = _uplift_controls(contract)
    else:
        effectiveness = params["effectiveness"]
        eff_key, simulating = ("uniform", params["effectiveness"]), False

    # KPIs and the customer list come from one call at the current budget — a
    # single knapsack solve on the channel tab. Only the curve needs the sweep.
    call = {
        "cost": cost_vector,
        "effectiveness": effectiveness,
        "margin_factor": params["margin_factor"],
        "H": params["horizon"],
    }
    mask, info = STRATEGY_FUNCS[recommended](p, m, params["budget"], **call)
    _, comparison_info = STRATEGY_FUNCS[comparison](p, m, params["budget"], **call)

    _kpi_row(
        info,
        comparison_info,
        recommended,
        comparison,
        ADVANTAGE_TITLE[cost_mode],
        simulated=simulating,
    )

    if cost_mode == "channel" and not simulating:
        st.caption(
            ":orange[**Assumption.**] This advantage is computed with a uniform "
            f"effectiveness of {params['effectiveness']:.2f} for everyone. "
            "Customers on long contracts (Two year) are cheap to reach and the "
            "optimiser leans on them, but they are also the least likely to be "
            "persuadable — they cannot leave in the first place. If their real "
            "effectiveness were lower, this advantage would shrink: an estimated "
            "band of roughly 9–34% depending on the assumption, **not a measured "
            "range**. Turn on the simulation above to explore it."
        )

    st.divider()

    with st.spinner("Computing the budget curve…"):
        curve = _budget_curve(
            p,
            m,
            cost_vector,
            effectiveness,
            cost_mode,
            cost_key,
            eff_key,
            params["margin_factor"],
            params["horizon"],
        )

    labels = {k: STRATEGY_LABELS[k] for k in CURVE_STRATEGIES[cost_mode]}
    dashes = dict(CURVE_DASHES[cost_mode])

    if simulating:
        # Ghost line: where the optimum sat before the assumption was changed,
        # so the cost of the hypothesis is visible rather than asserted.
        reference = _budget_curve(
            p,
            m,
            cost_vector,
            params["effectiveness"],
            cost_mode,
            cost_key,
            ("uniform", params["effectiveness"]),
            params["margin_factor"],
            params["horizon"],
        )
        reference = reference[reference.strategy == "optimal"].assign(
            strategy="optimal_reference"
        )
        curve = pd.concat([curve, reference], ignore_index=True)
        labels["optimal_reference"] = STRATEGY_LABELS["optimal_reference"]
        dashes["optimal_reference"] = "dot"

    st.plotly_chart(
        plot_value_curve_plotly(
            curve,
            labels,
            STRATEGY_COLORS,
            params["budget"],
            params["margin_factor"],
            dashes=dashes,
            subtitle=(
                "flat contact cost"
                if cost_mode == "flat"
                else "contact cost by channel"
            ),
        ),
        width="stretch",
    )

    # Broadcasts: a float under uniform effectiveness, an array under simulation.
    scalar = effectiveness * params["margin_factor"] * annuity_factor(params["horizon"])
    table = _customer_table(
        mask,
        p,
        m,
        ids,
        cost_vector,
        scalar,
        show_cost=(cost_mode == "channel"),
        effectiveness=effectiveness if simulating else None,
    )

    header = (
        f"##### {len(table):,} recommended customers · budget €{params['budget']:,.0f}"
    )
    if simulating:
        # One extra knapsack solve, only in simulation mode, to say how far the
        # hypothesis moved the plan. Without it the reordering is real but
        # invisible: the customers it drops sit at the bottom of the list.
        baseline_mask, _ = STRATEGY_FUNCS[recommended](
            p,
            m,
            params["budget"],
            cost=cost_vector,
            effectiveness=params["effectiveness"],
            margin_factor=params["margin_factor"],
            H=params["horizon"],
        )
        delta = int(mask.sum()) - int(baseline_mask.sum())
        dropped = int((baseline_mask & ~mask).sum())
        added = int((mask & ~baseline_mask).sum())
        header += (
            f"  ·  :orange[{delta:+,} vs uniform effectiveness]"
            f"  ({dropped:,} dropped, {added:,} added)"
        )
    st.markdown(header)

    if simulating:
        segments = pd.Series(np.asarray(contract)[mask]).value_counts()
        base_segments = pd.Series(np.asarray(contract)[baseline_mask]).value_counts()
        rows = " · ".join(
            f"{seg}: {int(segments.get(seg, 0)):,} "
            f"(was {int(base_segments.get(seg, 0)):,})"
            for seg in CONTRACT_SEGMENTS
        )
        st.caption(f":orange[**Plan composition under the assumption**] — {rows}")

    ranking_note = (
        "Ranked by net value, the quantity the optimiser maximises once "
        "effectiveness varies by segment."
        if simulating
        else "Ranked by priority (p × m)."
    )
    caption = (
        f"{ranking_note} **Recoverable value** is gross: the revenue "
        "expected back if the action works, before paying for the contact. "
        "**Net value** subtracts the contact cost — it is what the optimiser "
        "maximises, and it can be negative while the gross figure is positive."
    )
    if cost_mode == "channel":
        caption += (
            " **Contact cost** varies by contract here, which is why a cheap "
            "customer can be worth acting on ahead of a more valuable but "
            "expensive one."
        )
    st.caption(caption)
    st.dataframe(table, height=320, width="stretch", hide_index=True)
