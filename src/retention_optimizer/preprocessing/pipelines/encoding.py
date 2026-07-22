"""Encoding and split pipeline for the Telco churn dataset.

Turns the preprocessed dataframe (cleaning + feature engineering) into the
fully numeric matrix the models consume, and provides the canonical
stratified 80/20 split shared by every notebook:
    - Yes/No columns mapped to 1/0 (including the `Churn` target).
    - Multi-category columns one-hot encoded with `drop_first`.
    - Continuous columns optionally standardized (linear models need it;
      tree models are scale-invariant, so it is a separate opt-in step).
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Multi-category columns to one-hot encode. `tenure` is included because
# feature engineering rebins it into categorical churn-risk brackets.
ONE_HOT_COLS = ["gender", "InternetService", "Contract", "PaymentMethod", "tenure"]

# Continuous columns for `scale_numeric`. `SeniorCitizen` is excluded: it is
# a 0/1 flag that only looks numeric.
CONTINUOUS_COLS = ["MonthlyCharges", "TotalCharges", "num_services"]


def encode(d: pd.DataFrame) -> pd.DataFrame:
    """Turn the preprocessed dataframe into a fully numeric one.

    Drops `customerID` (an identifier, not a feature); callers that need it
    must keep it aside before encoding. The index is preserved, so the kept
    ids stay aligned with the encoded rows.
    """
    d = d.copy()
    if "customerID" in d.columns:
        d = d.drop(columns="customerID")
    binary_cols = [c for c in d.columns if set(d[c].dropna().unique()) == {"Yes", "No"}]
    d[binary_cols] = d[binary_cols].apply(lambda c: c.map({"Yes": 1, "No": 0}))
    d = pd.get_dummies(d, columns=ONE_HOT_COLS, drop_first=True, dtype=int)
    assert d.select_dtypes(exclude="number").columns.empty, "non-numeric columns left"
    return d


def split_data(
    d: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Canonical stratified split into `X_train, X_test, y_train, y_test`.

    Keep the defaults untouched across notebooks: the fixed seed guarantees
    every notebook sees the exact same train/test partition.
    """
    X = d.drop(columns="Churn")
    y = d["Churn"]
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


def scale_numeric(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Standardize the continuous columns, fitting the scaler on train only.

    Returns scaled copies plus the fitted scaler, needed to transform any
    future data the same way. Only linear models require this step: tree
    models are invariant to monotonic per-feature transforms.
    """
    X_train, X_test = X_train.copy(), X_test.copy()
    scaler = StandardScaler()
    X_train[CONTINUOUS_COLS] = scaler.fit_transform(X_train[CONTINUOUS_COLS])
    X_test[CONTINUOUS_COLS] = scaler.transform(X_test[CONTINUOUS_COLS])
    return X_train, X_test, scaler
