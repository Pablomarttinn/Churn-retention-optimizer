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
from retention_optimizer.preprocessing.pipelines.feature_engineering import (
    create_num_services,
    create_tenure_groups,
    engineer_features,
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
