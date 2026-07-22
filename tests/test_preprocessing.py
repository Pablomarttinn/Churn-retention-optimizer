"""Tests for the preprocessing pipelines (cleaning + feature engineering)."""

import numpy as np
import pandas as pd
import pytest

from retention_optimizer.preprocessing import preprocess
from retention_optimizer.preprocessing.pipelines.cleaning import (
    clean_data,
    collapse_no_service,
    fix_total_charges,
)
from retention_optimizer.preprocessing.pipelines.encoding import (
    CONTINUOUS_COLS,
    encode,
    scale_numeric,
    split_data,
)
from retention_optimizer.preprocessing.pipelines.feature_engineering import (
    create_num_services,
    create_tenure_groups,
)


@pytest.fixture
def raw() -> pd.DataFrame:
    """Minimal frame with the columns the pipelines touch.

    Row 0: brand-new customer (tenure 0, blank TotalCharges, no internet).
    Row 1: fiber customer with some add-on services.
    Row 2: long-tenure customer with every service.
    """
    return pd.DataFrame(
        {
            "tenure": [0, 12, 72],
            "TotalCharges": [" ", "820.5", "8300.1"],
            "MultipleLines": ["No phone service", "Yes", "No"],
            "PhoneService": ["No", "Yes", "Yes"],
            "OnlineSecurity": ["No internet service", "No", "Yes"],
            "OnlineBackup": ["No internet service", "Yes", "Yes"],
            "DeviceProtection": ["No internet service", "No", "Yes"],
            "TechSupport": ["No internet service", "No", "Yes"],
            "StreamingTV": ["No internet service", "Yes", "Yes"],
            "StreamingMovies": ["No internet service", "No", "Yes"],
        }
    )


# --- cleaning -------------------------------------------------------------


def test_fix_total_charges_casts_and_fills(raw):
    out = fix_total_charges(raw)
    assert out["TotalCharges"].dtype == float
    assert out["TotalCharges"].tolist() == [0.0, 820.5, 8300.1]


def test_collapse_no_service_removes_redundant_categories(raw):
    out = collapse_no_service(raw)
    # the "No * service" third category must be gone everywhere
    assert not (out == "No internet service").any().any()
    assert not (out == "No phone service").any().any()
    # it becomes plain "No"
    assert out.loc[0, "OnlineSecurity"] == "No"
    assert out.loc[0, "MultipleLines"] == "No"


def test_clean_data_runs_both_steps(raw):
    out = clean_data(raw)
    assert out["TotalCharges"].dtype == float
    assert not (out == "No internet service").any().any()


# --- feature engineering --------------------------------------------------


def test_create_num_services_counts_yes(raw):
    # count is done on raw "Yes" values, before collapsing
    out = create_num_services(raw)
    # row 0: only "No"/"No * service" -> 0 ; row 2: all 7 services "Yes"
    assert out.loc[0, "num_services"] == 0
    assert out.loc[2, "num_services"] == 7


def test_create_tenure_groups_bins_and_flags(raw):
    out = create_tenure_groups(raw)
    assert out["tenure"].tolist() == ["0-5", "6-17", "61-72"]
    assert out["new_customer"].tolist() == ["Yes", "No", "No"]


def test_create_tenure_groups_rejects_out_of_range():
    bad = pd.DataFrame({"tenure": [999]})
    with pytest.raises(AssertionError):
        create_tenure_groups(bad)


# --- encoding & split -------------------------------------------------------


@pytest.fixture
def raw_full() -> pd.DataFrame:
    """Full-schema frame (all raw Telco columns), same three customers as `raw`."""
    return pd.DataFrame(
        {
            "customerID": ["0001-AAAAA", "0002-BBBBB", "0003-CCCCC"],
            "gender": ["Female", "Male", "Female"],
            "SeniorCitizen": [0, 0, 1],
            "Partner": ["No", "Yes", "Yes"],
            "Dependents": ["No", "No", "Yes"],
            "tenure": [0, 12, 72],
            "PhoneService": ["No", "Yes", "Yes"],
            "MultipleLines": ["No phone service", "Yes", "No"],
            "InternetService": ["No", "Fiber optic", "DSL"],
            "OnlineSecurity": ["No internet service", "No", "Yes"],
            "OnlineBackup": ["No internet service", "Yes", "Yes"],
            "DeviceProtection": ["No internet service", "No", "Yes"],
            "TechSupport": ["No internet service", "No", "Yes"],
            "StreamingTV": ["No internet service", "Yes", "Yes"],
            "StreamingMovies": ["No internet service", "No", "Yes"],
            "Contract": ["Month-to-month", "Month-to-month", "Two year"],
            "PaperlessBilling": ["Yes", "Yes", "No"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Credit card (automatic)",
            ],
            "MonthlyCharges": [20.0, 85.5, 110.2],
            "TotalCharges": [" ", "820.5", "8300.1"],
            "Churn": ["Yes", "No", "No"],
        }
    )


def test_encode_returns_fully_numeric_frame(raw_full):
    out = encode(preprocess(raw_full))
    assert out.select_dtypes(exclude="number").columns.empty
    assert "customerID" not in out.columns


def test_encode_maps_yes_no_and_preserves_index(raw_full):
    df = preprocess(raw_full).set_axis([10, 20, 30], axis="index")
    out = encode(df)
    # Yes/No columns (target included) become 1/0
    assert out["Churn"].tolist() == [1, 0, 0]
    assert out["Partner"].tolist() == [0, 1, 1]
    # index survives, so a customerID kept aside still aligns
    assert out.index.tolist() == [10, 20, 30]


def test_split_data_is_stratified_and_reproducible():
    df = pd.DataFrame({"a": range(20), "Churn": [0, 1] * 10})
    X_train, X_test, y_train, y_test = split_data(df)
    assert len(X_train) == 16 and len(X_test) == 4
    assert y_train.mean() == y_test.mean() == 0.5
    # same seed -> exact same partition
    X_train2, _, _, _ = split_data(df)
    pd.testing.assert_frame_equal(X_train, X_train2)


def test_scale_numeric_fits_on_train_only():
    X_train = pd.DataFrame(
        {
            "MonthlyCharges": [10.0, 20.0],
            "TotalCharges": [1.0, 3.0],
            "num_services": [0, 4],
        }
    )
    X_test = X_train + 100  # deliberately shifted
    train_before, test_before = X_train.copy(), X_test.copy()
    out_train, out_test, scaler = scale_numeric(X_train, X_test)
    # train is centered; the shifted test is not (scaler never saw it)
    assert np.allclose(out_train[CONTINUOUS_COLS].mean(), 0)
    assert (out_test[CONTINUOUS_COLS].mean() > 1).all()
    # inputs are returned as copies, not mutated
    pd.testing.assert_frame_equal(X_train, train_before)
    pd.testing.assert_frame_equal(X_test, test_before)


# --- orchestration & purity ----------------------------------------------


def test_preprocess_end_to_end(raw):
    out = preprocess(raw)
    # both cleaning and FE happened
    assert out["TotalCharges"].dtype == float
    assert "num_services" in out.columns
    assert "new_customer" in out.columns
    assert out["tenure"].tolist() == ["0-5", "6-17", "61-72"]


def test_pipelines_do_not_mutate_input(raw):
    before = raw.copy()
    _ = preprocess(raw)
    # the .copy() inside each step must leave the caller's frame untouched
    pd.testing.assert_frame_equal(raw, before)
