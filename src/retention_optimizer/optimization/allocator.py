"""Retention budget allocation as a 0/1 knapsack.

Given a per-customer net value and a per-customer action cost, pick the subset
of customers that maximizes total net value without exceeding the budget. The
selection is delegated entirely to the MILP solver — no manual sorting or
cut-off — so heterogeneous costs are handled correctly.
"""

import numpy as np
from pulp import LpBinary, LpMaximize, LpProblem, LpSolverDefault, lpSum

from retention_optimizer.optimization.value import expected_value

# Use PuLP's bundled CBC (the implicit default solver) rather than an external
# COIN_CMD binary: it ships with the `pulp` wheel, so anyone who runs `uv sync`
# gets the exact same solver and the results stay reproducible.
LpSolverDefault.msg = 0


def solve_knapsack(
    net_value: np.ndarray,
    cost: np.ndarray,
    budget: float,
) -> np.ndarray:
    """Solve the 0/1 knapsack over customers.

    Maximizes ``sum(x_i * net_value_i)`` subject to
    ``sum(x_i * cost_i) <= budget`` with ``x_i`` binary.

    Args:
        net_value: Net value per customer; may be negative (a negative entry is
            simply never worth selecting).
        cost: Action cost per customer, same length as `net_value`.
        budget: Total budget available.

    Returns:
        Boolean mask, True where the customer is selected.

    Raises:
        RuntimeError: If the solver does not reach an optimal solution.
    """
    n = len(net_value)
    indices = range(n)

    prob = LpProblem("retention_knapsack", LpMaximize)
    x = prob.add_variable_dicts("x", indices, cat=LpBinary)

    prob += lpSum(x[i] * float(net_value[i]) for i in indices)
    prob += lpSum(x[i] * float(cost[i]) for i in indices) <= budget

    prob.solve()
    if prob.status != 1:
        raise RuntimeError(
            f"Knapsack solver did not find an optimum (status={prob.status})"
        )

    return np.array([x[i].value() > 0.5 for i in indices], dtype=bool)


def allocate(
    p: np.ndarray,
    m: np.ndarray,
    budget: float,
    cost: float | np.ndarray = 20.0,
    effectiveness: float | np.ndarray = 0.3,
    margin_factor: float = 1.0,
    H: int = 12,
    d: float = 0.01,
) -> tuple[np.ndarray, dict]:
    """Allocate the retention budget across customers.

    Computes the expected value of acting on each customer, nets out the action
    cost, and hands the result to the knapsack solver.

    `cost` and `effectiveness` accept a scalar or a per-customer vector: both
    are broadcast to the shape of `p` up front, so per-segment costs or
    per-segment action effectiveness need no change to the call site.

    Args:
        p: Calibrated churn probability per customer.
        m: Monthly margin per customer.
        budget: Total retention budget in euros.
        cost: Cost of the retention action, scalar or per customer.
        effectiveness: Share of targeted churners saved, scalar or per customer.
        margin_factor: Fraction of revenue that is margin.
        H: Horizon in months of retained margin.
        d: Monthly discount rate.

    Returns:
        Tuple of (mask, info): the boolean selection mask, and a dict with the
        realized spend, the expected value captured, and how many customers
        were selected.
    """
    p = np.asarray(p, dtype=float)
    m = np.asarray(m, dtype=float)
    cost = np.broadcast_to(np.asarray(cost, dtype=float), p.shape)
    effectiveness = np.broadcast_to(np.asarray(effectiveness, dtype=float), p.shape)

    ev = expected_value(p, m, effectiveness, margin_factor, H, d)
    net = ev - cost

    mask = solve_knapsack(net, cost, budget)

    info = {
        "spend": float((cost * mask).sum()),
        "expected_value": float((ev * mask).sum()),
        "n_selected": int(mask.sum()),
    }
    return mask, info
