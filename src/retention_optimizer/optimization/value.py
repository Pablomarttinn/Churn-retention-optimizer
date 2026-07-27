"""Economic value primitives for the retention budget optimizer.

All functions are vectorized over numpy arrays: they accept plain scalars or
numpy arrays indistinctly and never touch DataFrames internally. Callers are
responsible for extracting the columns they need (e.g. `df["MonthlyCharges"]
.to_numpy()`) before calling in.

Monetary units are euros; `d` is always a *monthly* discount rate.
"""

import numpy as np

Numeric = float | np.ndarray


def annuity_factor(H: int, d: float = 0.01) -> float:
    """Present value of 1 unit of margin received at the end of each of H months.

    Closed form (ordinary annuity)::

        a(H, d) = (1 - (1 + d) ** -H) / d

    Args:
        H: Horizon in months over which the retained margin is counted.
        d: Monthly discount rate (0.01 == 1% per month).

    Returns:
        The annuity factor, i.e. how many months of margin the horizon is worth
        once discounted. Always smaller than `H`.
    """
    return (1 - (1 + d) ** -H) / d


def expected_value(
    p: Numeric,
    m: Numeric,
    effectiveness: float = 0.3,
    margin_factor: float = 1.0,
    H: int = 12,
    d: float = 0.01,
) -> Numeric:
    """Expected euro value of running the retention action on a customer.

    Formula::

        scalar = effectiveness * margin_factor * annuity_factor(H, d)
        value  = p * m * scalar

    Note:
        `p * m` is the `priority_score` used for ranking. Everything else is a
        constant scalar that only rescales the euro amount — it never reorders
        the customer list, so ranking by `p * m` and ranking by
        `expected_value` are equivalent.

    Args:
        p: Calibrated churn probability P(churn) of the customer.
        m: Monthly margin of the customer (`MonthlyCharges` at this stage).
        effectiveness: Share of targeted churners the action actually saves.
        margin_factor: Fraction of revenue that is margin (1.0 == revenue).
        H: Horizon in months of retained margin.
        d: Monthly discount rate.

    Returns:
        Expected euros recovered by acting on the customer.
    """
    scalar = effectiveness * margin_factor * annuity_factor(H, d)
    return p * m * scalar
