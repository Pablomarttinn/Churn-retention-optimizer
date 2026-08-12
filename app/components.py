"""The decision view: sliders, KPIs, curve and recommended-customer list.

`render_decision_view` is parameterised by cost regime rather than written for a
particular screen. Screen 1 instantiates it with a flat contact cost; screen 2
will instantiate the same component with a per-channel cost vector and the
knapsack as the recommended strategy, without this file changing shape.
"""

import numpy as np
import pandas as pd
import streamlit as st

from app.plotting import plot_value_curve_plotly
from app.theme import ACCENT, NEUTRAL, STRATEGY_COLORS, STRATEGY_LABELS
from retention_optimizer.evaluation.strategies import (
    strategy_by_churn,
    strategy_by_priority,
    strategy_optimal,
)
from retention_optimizer.evaluation.sweep import sweep_strategies
from retention_optimizer.optimization.value import annuity_factor

# Coarser than the notebook sweep (step 1000): the curve is redrawn on every
# slider move, and 40 points already render as a smooth spline.
BUDGET_GRID = np.arange(1000, 80001, 2000)

# Which strategy the view recommends, per cost regime. With a flat cost the
# ranking by p*m is already optimal; once costs differ the knapsack is not.
RECOMMENDED_STRATEGY = {"flat": "by_priority", "channel": "optimal"}
BASELINE_STRATEGY = "by_churn"

STRATEGY_FUNCS = {
    "by_churn": strategy_by_churn,
    "by_priority": strategy_by_priority,
    "optimal": strategy_optimal,
}


@st.cache_data(show_spinner=False)
def _budget_curve(
    _p: np.ndarray,
    _m: np.ndarray,
    _cost,
    cost_key: float,
    recommended: str,
    effectiveness: float,
    margin_factor: float,
    horizon: int,
) -> pd.DataFrame:
    """Cached budget sweep for the two lines on the chart.

    The leading-underscore arguments are excluded from the cache key (the arrays
    never change within a session); `cost_key` carries the part of the cost that
    does vary, so moving the cost slider invalidates the entry but re-rendering
    with untouched sliders does not.
    """
    strategies = {
        BASELINE_STRATEGY: STRATEGY_FUNCS[BASELINE_STRATEGY],
        recommended: STRATEGY_FUNCS[recommended],
    }
    return sweep_strategies(
        _p,
        _m,
        BUDGET_GRID,
        strategies,
        cost=_cost,
        effectiveness=effectiveness,
        margin_factor=margin_factor,
        H=horizon,
    )


def _sliders() -> dict:
    """Sidebar controls. Every label carries its unit."""
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
        "Cost per contact (EUR)",
        min_value=5,
        max_value=50,
        value=20,
        step=1,
        help="What one retention action costs for a single customer.",
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

    return {
        "budget": float(budget),
        "cost": float(cost),
        "effectiveness": float(effectiveness),
        "margin_factor": float(margin_factor),
        "horizon": int(horizon),
    }


def _kpi_row(recommended_info: dict, baseline_info: dict, recommended: str) -> None:
    """Five metrics, each stating its unit.

    The headline advantage is shown in euros *and* as a share of the baseline,
    together on purpose: the euro gap widens as the budget grows, while the
    relative gap is largest when the budget is tightest. Either number on its
    own tells half the story and points at the opposite conclusion about where
    prioritising matters most.
    """
    net_value = recommended_info["expected_value"] - recommended_info["spend"]
    baseline_net = baseline_info["expected_value"] - baseline_info["spend"]
    roi = recommended_info["expected_value"] / recommended_info["spend"]

    advantage = net_value - baseline_net
    # The baseline can retain nothing at all under harsh slider settings (a high
    # contact cost with a low margin leaves nobody profitable), and a percentage
    # off a zero base would be meaningless rather than merely large.
    advantage_pct = advantage / baseline_net * 100 if baseline_net > 0 else None
    pct_text = f"({advantage_pct:.1f}%)" if advantage_pct is not None else "(n/a)"

    ours = STRATEGY_LABELS[recommended]
    theirs = STRATEGY_LABELS[BASELINE_STRATEGY]
    formula = (
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

    c5.markdown(
        f"""
        <div style="padding-top:2px">
          <div style="color:{NEUTRAL};font-size:0.8rem">
            Advantage vs risk targeting
          </div>
          <div style="color:{ACCENT};font-size:2.1rem;font-weight:700;
                      line-height:1.25">
            +€{advantage:,.0f}
            <span style="font-size:1.35rem;font-weight:600">{pct_text}</span>
          </div>
          <div style="color:{NEUTRAL};font-size:0.7rem;line-height:1.35;
                      padding-top:2px">{formula}</div>
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
    scalar: float,
) -> pd.DataFrame:
    """Recommended customers, ranked by priority, with gross and net value."""
    recoverable = p * m * scalar
    table = pd.DataFrame(
        {
            "customerID": np.asarray(ids)[mask],
            "Churn risk (%)": (p[mask] * 100).round(1),
            "Monthly value (EUR/month)": m[mask].round(2),
            "Priority (p × m)": (p[mask] * m[mask]).round(2),
            "Recoverable value (EUR) — gross": recoverable[mask].round(2),
            "Net value (EUR) — after contact cost": (recoverable - cost)[mask].round(2),
        }
    )
    return table.sort_values("Priority (p × m)", ascending=False).reset_index(drop=True)


def render_decision_view(
    cost_mode: str,
    p: np.ndarray,
    m: np.ndarray,
    contract: pd.Series,
    ids: pd.Series,
) -> None:
    """Render the whole decision screen for one cost regime.

    Args:
        cost_mode: `"flat"` (one cost for everyone, driven by the slider) or
            `"channel"` (a per-customer cost vector derived from `contract`).
        p: Calibrated churn probability per customer.
        m: Monthly margin per customer.
        contract: The raw `Contract` column, aligned to `ids`. Unused under
            `"flat"`; the channel regime derives its cost vector from it.
        ids: Customer identifiers, aligned positionally with `p` and `m`.
    """
    if cost_mode not in RECOMMENDED_STRATEGY:
        raise ValueError(f"unknown cost_mode: {cost_mode!r}")

    params = _sliders()
    recommended = RECOMMENDED_STRATEGY[cost_mode]

    if cost_mode == "flat":
        cost_vector = np.full(len(p), params["cost"])
        cost_key = params["cost"]
        regime_caption = (
            f"Flat contact cost of €{params['cost']:,.0f} for every customer"
        )
    else:
        # Screen 2 wires the per-channel cost vector in here; the rest of this
        # function already handles a vector cost without changes.
        raise NotImplementedError(
            "channel cost regime is prepared but not implemented yet"
        )

    st.markdown(f"#### {STRATEGY_LABELS[recommended]} — recommended plan")
    st.caption(regime_caption)

    # KPIs and the customer list come from a single call at the current budget,
    # which is instant; only the curve needs the full sweep.
    call = {
        "cost": cost_vector,
        "effectiveness": params["effectiveness"],
        "margin_factor": params["margin_factor"],
        "H": params["horizon"],
    }
    mask, info = STRATEGY_FUNCS[recommended](p, m, params["budget"], **call)
    _, baseline_info = STRATEGY_FUNCS[BASELINE_STRATEGY](p, m, params["budget"], **call)

    _kpi_row(info, baseline_info, recommended)
    st.divider()

    curve = _budget_curve(
        p,
        m,
        cost_vector,
        cost_key,
        recommended,
        params["effectiveness"],
        params["margin_factor"],
        params["horizon"],
    )
    labels = {k: STRATEGY_LABELS[k] for k in (BASELINE_STRATEGY, recommended)}
    st.plotly_chart(
        plot_value_curve_plotly(
            curve,
            labels,
            STRATEGY_COLORS,
            params["budget"],
            params["margin_factor"],
        ),
        width="stretch",
    )

    scalar = (
        params["effectiveness"]
        * params["margin_factor"]
        * annuity_factor(params["horizon"])
    )
    table = _customer_table(mask, p, m, ids, cost_vector, scalar)

    st.markdown(
        f"##### {len(table):,} recommended customers · budget €{params['budget']:,.0f}"
    )
    st.caption(
        "Ranked by priority (p × m). **Recoverable value** is gross: the revenue "
        "expected back if the action works, before paying for the contact. "
        "**Net value** subtracts the contact cost — it is what the optimiser "
        "maximises, and it can be negative while the gross figure is positive."
    )
    st.dataframe(table, height=320, width="stretch", hide_index=True)
