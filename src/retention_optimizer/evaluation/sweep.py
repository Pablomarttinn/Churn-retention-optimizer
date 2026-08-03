"""Budget sweeps: evaluate every strategy across a range of budgets.

The sweep logic lives here rather than in a notebook because the dashboard
reuses it: a notebook orchestrates and plots, this module produces the numbers.
"""

import numpy as np
import pandas as pd

from retention_optimizer.evaluation.strategies import strategy_random


def load_oof(
    path: str = "data/processed/oof_predictions.csv",
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """Load the out-of-fold scores as the arrays the strategies consume.

    Row order is preserved exactly as stored in the CSV, which is the order the
    OOF were generated in. Every downstream array — `p`, `m`, a cost vector, a
    selection mask — is aligned to it positionally, so `ids` is what any later
    join back to customer-level data must go through.

    Returns:
        Tuple of (p, m, ids): churn probability, monthly margin, and the
        customer identifiers, all in the same row order.
    """
    df = pd.read_csv(path)
    return (
        df["p_oof"].to_numpy(),
        df["MonthlyCharges"].to_numpy(),
        df["customerID"],
    )


def random_expected_value(
    p: np.ndarray,
    m: np.ndarray,
    budget: float,
    cost: float | np.ndarray = 20.0,
    n_seeds: int = 30,
    **kwargs,
) -> float:
    """Mean expected value of the random strategy over `n_seeds` draws.

    A single random draw is an anecdote: it can land well or badly and the
    chart would inherit that luck. The comparison floor has to be the expected
    value of acting at random, so the seeds 0..n_seeds-1 are averaged.

    Returns:
        The mean `expected_value` in euros across the draws.
    """
    values = [
        strategy_random(p, m, budget, cost=cost, seed=s, **kwargs)[1]["expected_value"]
        for s in range(n_seeds)
    ]
    return float(np.mean(values))


def sweep_strategies(
    p: np.ndarray,
    m: np.ndarray,
    budgets: np.ndarray,
    strategies_dict: dict,
    cost: float | np.ndarray = 20.0,
    n_seeds: int = 30,
    **kwargs,
) -> pd.DataFrame:
    """Evaluate every strategy at every budget.

    The strategy named `"random"` is special-cased: its `expected_value` is the
    average over `n_seeds` draws, while `spend` and `n_selected` come from the
    single `seed=0` call — those two barely move between draws, since the
    budget and the flat cost fix how many customers fit.

    Args:
        p: Churn probability per customer.
        m: Monthly margin per customer.
        budgets: Budgets to evaluate, in euros.
        strategies_dict: Mapping of label to strategy function.
        cost: Action cost, scalar or per customer.
        n_seeds: Draws to average for the random strategy.
        **kwargs: Valuation parameters forwarded to every strategy
            (`effectiveness`, `margin_factor`, `H`, `d`).

    Returns:
        Long-format DataFrame with one row per budget x strategy and the
        columns `budget`, `strategy`, `expected_value`, `spend`, `n_selected`.
    """
    rows = []
    for budget in budgets:
        for name, strategy in strategies_dict.items():
            if name == "random":
                _, info = strategy(p, m, budget, cost=cost, seed=0, **kwargs)
                value = random_expected_value(
                    p, m, budget, cost=cost, n_seeds=n_seeds, **kwargs
                )
            else:
                _, info = strategy(p, m, budget, cost=cost, **kwargs)
                value = info["expected_value"]

            rows.append(
                {
                    "budget": float(budget),
                    "strategy": name,
                    "expected_value": value,
                    "spend": info["spend"],
                    "n_selected": info["n_selected"],
                }
            )

    return pd.DataFrame(rows)
