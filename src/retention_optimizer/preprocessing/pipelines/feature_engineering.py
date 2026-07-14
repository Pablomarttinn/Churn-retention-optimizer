"""Feature engineering pipeline for the Telco churn dataset.

Creates the features designed during EDA:
    - `num_services`  : how many services each customer subscribes to (stickiness).
    - `new_customer`  : flag for `tenure == 0`, where churn spikes.
    - `tenure`        : rebinned into churn-risk brackets (non-linear effect).
"""

import numpy as np
import pandas as pd

# Services counted towards `num_services`.
SERVICES = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "PhoneService",
]

# Tenure brackets, chosen from the observed (non-linear) churn-rate decay.
TENURE_BINS = [-1, 5, 17, 40, 60, 72]
TENURE_LABELS = ["0-5", "6-17", "18-40", "41-60", "61-72"]


def create_num_services(d: pd.DataFrame) -> pd.DataFrame:
    """Add `num_services`: the number of services each customer subscribes to."""
    d = d.copy()
    d["num_services"] = (d[SERVICES] == "Yes").sum(axis=1)
    return d


def create_tenure_groups(d: pd.DataFrame) -> pd.DataFrame:
    """Bin `tenure` into churn-risk brackets and flag brand-new customers.

    Brackets (observed churn rate):
        0-5     (54.3%)  initial cliff
        6-17    (35.2%)  steep decline, above the mean
        18-40   (22.8%)  plateau around the mean
        41-60   (15.6%)  second step down
        61-72   ( 6.6%)  loyal long-tenure customers
    """
    d = d.copy()
    d["new_customer"] = np.where(d["tenure"] == 0, "Yes", "No")
    d["tenure"] = pd.cut(
        d["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS, ordered=True
    )
    assert d["tenure"].notna().all(), "tenure values fall outside the defined bins"
    return d


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature-engineering pipeline.

    `create_num_services` runs first because it needs the raw service columns;
    `create_tenure_groups` overwrites `tenure`, so anything depending on the
    numeric tenure must run before it.
    """
    return df.pipe(create_num_services).pipe(create_tenure_groups)
