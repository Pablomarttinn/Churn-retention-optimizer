"""Budget allocation strategies with a homogeneous, interchangeable signature.

Every strategy answers the same question — which customers do we act on under a
given budget — and they all share the exact same signature and return shape, so
a caller can hold them in a list and loop over them. Adding a fifth strategy
means adding a function, not touching the four below or their consumers.

The split that matters:
    - The **selection criterion** is what changes between strategies (random,
      churn probability, priority score, MILP optimum).
    - The **euro valuation** of whatever was selected never changes: it always
      goes through `value.expected_value` with the same parameters. That is
      what makes the four comparable — a strategy cannot look better because it
      valued its own picks differently.
"""

import numpy as np
import pandas as pd

from retention_optimizer.optimization.allocator import allocate
from retention_optimizer.optimization.value import expected_value

Numeric = float | np.ndarray


def _prepare(
    p: np.ndarray, m: np.ndarray, cost: Numeric, effectiveness: Numeric
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Coerce inputs to arrays and broadcast the scalar-or-vector parameters."""
    p = np.asarray(p, dtype=float)
    m = np.asarray(m, dtype=float)
    cost = np.broadcast_to(np.asarray(cost, dtype=float), p.shape)
    effectiveness = np.broadcast_to(np.asarray(effectiveness, dtype=float), p.shape)
    return p, m, cost, effectiveness


def _summarize(
    mask: np.ndarray,
    p: np.ndarray,
    m: np.ndarray,
    cost: np.ndarray,
    effectiveness: np.ndarray,
    margin_factor: float,
    H: int,
    d: float,
) -> dict:
    """Value a selection in euros, with the same formula for every strategy.

    Deliberately recomputed here even for `strategy_optimal`, which gets an
    equivalent dict back from `allocate`: routing all four through one function
    means the comparison stays honest by construction, not by coincidence.
    """
    ev = expected_value(p, m, effectiveness, margin_factor, H, d)
    return {
        "spend": float((cost * mask).sum()),
        "expected_value": float((ev * mask).sum()),
        "n_selected": int(mask.sum()),
    }


def _greedy_take(
    order: np.ndarray,
    net_value: np.ndarray | None,
    cost: np.ndarray,
    budget: float,
) -> np.ndarray:
    """Walk an ordered list of customers, taking the ones worth paying for.

    Two independent stopping rules, and they behave differently on purpose:

    - **Negative net value: skip, do not stop.** A customer whose action costs
      more than it recovers is never worth contacting, however high they rank.
      Net value is not monotone in the `p` or `p * m` ordering, so a cheap
      profitable customer can sit below an unprofitable one; breaking at the
      first loss would silently discard them.
    - **Budget exhausted: stop.** No skipping ahead in search of someone
      cheaper who would still fit. That is what keeps these strategies naive
      rankings and leaves the packing gains to the knapsack.

    Args:
        order: Customer indices in the strategy's ranking order.
        net_value: Per-customer expected value minus its action cost, computed
            with the same valuation parameters as the calling strategy, so the
            profitability threshold moves with those parameters instead of
            being fixed. `None` disables the filter entirely — used by the
            random floor, which is meant to know nothing, profitability
            included.
        cost: Per-customer action cost.
        budget: Total budget available.

    Returns:
        Boolean mask of the selected customers.
    """
    mask = np.zeros(len(cost), dtype=bool)
    spent = 0.0
    for i in order:
        if net_value is not None and net_value[i] <= 0:
            continue
        c = float(cost[i])
        if spent + c > budget:
            break
        mask[i] = True
        spent += c
    return mask


def strategy_random(
    p: np.ndarray,
    m: np.ndarray,
    budget: float,
    cost: Numeric = 20.0,
    effectiveness: Numeric = 0.3,
    margin_factor: float = 1.0,
    H: int = 12,
    d: float = 0.01,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """Criterion: none — customers are drawn at random until the budget runs out.

    The comparison floor. Any strategy that fails to beat it is not adding
    information, and its margin over this one is what the model is worth.

    Unlike the ranked strategies, this one keeps contacting customers whose net
    value is negative. That is the point: the floor must know *nothing*, and
    filtering on profitability would hand it the single most valuable piece of
    information in the problem. Consequence to expect — past the profitability
    elbow its value curve turns downwards, which is a true statement about
    spending at random, not a defect.
    """
    p, m, cost, effectiveness = _prepare(p, m, cost, effectiveness)
    order = np.random.default_rng(seed).permutation(len(p))
    mask = _greedy_take(order, None, cost, budget)
    return mask, _summarize(mask, p, m, cost, effectiveness, margin_factor, H, d)


def strategy_by_churn(
    p: np.ndarray,
    m: np.ndarray,
    budget: float,
    cost: Numeric = 20.0,
    effectiveness: Numeric = 0.3,
    margin_factor: float = 1.0,
    H: int = 12,
    d: float = 0.01,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """Criterion: highest churn probability first, cut off by the budget.

    The serious baseline — targeting whoever is most likely to leave is
    standard industry practice. It ignores how much each customer is worth.
    """
    p, m, cost, effectiveness = _prepare(p, m, cost, effectiveness)
    order = np.argsort(p)[::-1]
    net = expected_value(p, m, effectiveness, margin_factor, H, d) - cost
    mask = _greedy_take(order, net, cost, budget)
    return mask, _summarize(mask, p, m, cost, effectiveness, margin_factor, H, d)


def strategy_by_priority(
    p: np.ndarray,
    m: np.ndarray,
    budget: float,
    cost: Numeric = 20.0,
    effectiveness: Numeric = 0.3,
    margin_factor: float = 1.0,
    H: int = 12,
    d: float = 0.01,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """Criterion: highest `priority_score = p * m` first, cut off by the budget.

    The method this project proposes: weight the churn risk by what the
    customer is actually worth. Still a ranking, so it does not optimize
    against a heterogeneous cost structure.
    """
    p, m, cost, effectiveness = _prepare(p, m, cost, effectiveness)
    order = np.argsort(p * m)[::-1]
    net = expected_value(p, m, effectiveness, margin_factor, H, d) - cost
    mask = _greedy_take(order, net, cost, budget)
    return mask, _summarize(mask, p, m, cost, effectiveness, margin_factor, H, d)


def strategy_optimal(
    p: np.ndarray,
    m: np.ndarray,
    budget: float,
    cost: Numeric = 20.0,
    effectiveness: Numeric = 0.3,
    margin_factor: float = 1.0,
    H: int = 12,
    d: float = 0.01,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """Criterion: the 0/1 knapsack optimum, solved as a MILP.

    With a constant cost this collapses to `strategy_by_priority` — the budget
    just buys a fixed number of slots, so the top-k of the ranking is already
    optimal. Its advantage only materializes when `cost` is a heterogeneous
    vector and the packing decision stops being trivial.

    `seed` is unused: the solver is deterministic. It stays in the signature to
    keep the four strategies interchangeable.
    """
    p, m, cost, effectiveness = _prepare(p, m, cost, effectiveness)
    mask, _ = allocate(
        p,
        m,
        budget=budget,
        cost=cost,
        effectiveness=effectiveness,
        margin_factor=margin_factor,
        H=H,
        d=d,
    )
    return mask, _summarize(mask, p, m, cost, effectiveness, margin_factor, H, d)


def channel_cost_from_contract(
    contract_series: pd.Series,
    costs: dict | None = None,
) -> np.ndarray:
    """Map the `Contract` column to a per-customer retention cost.

    This is a **declared business policy**, not an estimate learned from data:
    reaching a month-to-month customer needs an expensive channel (a call, a
    discount worth defending), while a two-year contract customer can be
    handled by a cheap automated touch. The policy is expressed over an
    observable variable already in the dataset, so the resulting cost vector is
    reproducible from the raw data rather than invented per run.

    This is what turns the strategy comparison into a real test: with a flat
    cost the knapsack has nothing to optimize.

    Args:
        contract_series: The raw `Contract` column.
        costs: Policy override; defaults to the declared channel prices.
    """
    if costs is None:
        costs = {
            "Month-to-month": 40.0,
            "One year": 15.0,
            "Two year": 3.0,
        }

    mapped = contract_series.map(costs)
    assert mapped.notna().all(), (
        f"unmapped Contract values: "
        f"{sorted(set(contract_series[mapped.isna()].unique()))}"
    )
    return mapped.to_numpy(dtype=float)
