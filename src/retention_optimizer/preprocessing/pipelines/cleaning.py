"""Data cleaning pipeline for the Telco churn dataset.

Fixes data-quality issues found during EDA:
    - `TotalCharges` stored as a string with blank values for new customers.
    - Redundant "No internet service" / "No phone service" categories that are
      already captured by `InternetService` / `PhoneService`.
"""

import pandas as pd

# Service columns whose "No internet service" third category is redundant:
# it is fully captured by the `InternetService` column.
INTERNET_SERVICES = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def fix_total_charges(d: pd.DataFrame) -> pd.DataFrame:
    """Cast `TotalCharges` to float and fill the blanks with 0.

    The 11 blank rows correspond to new customers (`tenure == 0`) who have not
    been billed yet, so 0 is the correct value.
    """
    d = d.copy()
    d["TotalCharges"] = pd.to_numeric(d["TotalCharges"], errors="coerce").fillna(0)
    return d


def collapse_no_service(d: pd.DataFrame) -> pd.DataFrame:
    """Collapse the redundant "No * service" categories into "No".

    The "has internet / phone" signal is kept in `InternetService` /
    `PhoneService`, so the model can still reconstruct the original three cases
    without repeating that information in every service column.
    """
    d = d.copy()
    d[INTERNET_SERVICES] = d[INTERNET_SERVICES].replace("No internet service", "No")
    d["MultipleLines"] = d["MultipleLines"].replace("No phone service", "No")
    return d


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline."""
    return df.pipe(fix_total_charges).pipe(collapse_no_service)
