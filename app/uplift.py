"""Simulated heterogeneous effectiveness, by contract segment.

**Nothing here is measured.** `channel_cost_from_contract` in the strategy module
maps a *declared business policy* — what the company chooses to spend per
channel — onto an observable column. This is its twin in shape only: it maps a
*hypothesis the user is entertaining* onto the same column.

How persuadable a customer actually is cannot be read off historical data. The
observed churn probability says who is likely to leave, not who could be talked
out of it, and separating the two needs a randomised experiment. Everything this
module produces is therefore a what-if, and every surface that consumes it has to
say so.

This lives in `app/` rather than in `retention_optimizer` on purpose: the package
holds what the project can defend, and a slider-driven assumption is not that.
"""

import numpy as np
import pandas as pd

# Same three segments the channel cost policy is declared over.
CONTRACT_SEGMENTS = ("Month-to-month", "One year", "Two year")

# The value the uniform (non-simulated) mode uses. The per-segment sliders start
# here so that switching simulation on, untouched, reproduces the real-data run
# exactly — any divergence is the user's explicit choice, never the app's.
UNIFORM_EFFECTIVENESS = 0.30


def effectiveness_from_contract(contract: pd.Series, effs: dict) -> np.ndarray:
    """Map each customer to the assumed effectiveness of their contract segment.

    Args:
        contract: The raw `Contract` column, one row per customer.
        effs: Assumed effectiveness per segment, e.g.
            `{"Month-to-month": 0.30, "One year": 0.30, "Two year": 0.05}`.
            These are **user assumptions being explored**, not estimates: no
            number in this project measures per-segment persuadability.

    Returns:
        Per-customer effectiveness, aligned positionally with `contract`.

    Raises:
        AssertionError: If a contract value has no assumed effectiveness, which
            would otherwise propagate as a silent NaN into the optimiser.
    """
    mapped = contract.map(effs)
    assert mapped.notna().all(), (
        "contract values with no assumed effectiveness: "
        f"{sorted(set(contract[mapped.isna()].unique()))}"
    )
    return mapped.to_numpy(dtype=float)
