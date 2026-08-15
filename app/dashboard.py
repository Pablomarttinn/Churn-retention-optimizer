"""Retention Budget Optimizer — decision dashboard.

Run with:
    poetry run streamlit run app/dashboard.py
    (this repo uses uv: `uv run streamlit run app/dashboard.py`)
"""

import sys
from pathlib import Path

# `streamlit run` puts this file's own directory on sys.path, not the repo root,
# so `import app.…` would fail. Prepend the root before the local imports below.
#
# `src` goes on the path too, so `retention_optimizer` imports without the
# package being installed. That is what lets the deployment skip `pip install
# -e .`, which would drag in every dependency declared in pyproject.toml —
# xgboost, scikit-learn, shap, jupyter — none of which this app runs.
ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.components import render_decision_view, sidebar_controls  # noqa: E402
from app.theme import BG, CARD_BG, NEUTRAL, TEXT  # noqa: E402

# The one file the app reads at runtime. Everything expensive — preprocess(),
# and with it scikit-learn, xgboost and joblib — happens offline in
# scripts/build_dashboard_data.py, so the deployed container needs neither the
# raw dataset nor the modelling stack.
DATA_PATH = ROOT / "data" / "processed" / "dashboard_data.csv"

st.set_page_config(
    layout="wide",
    page_title="Retention Budget Optimizer",
    page_icon="◈",
)

# Minimal styling: background and legibility only. Visual polish is a later step.
st.markdown(
    f"""
    <style>
      .stApp {{ background-color: {BG}; color: {TEXT}; }}
      section[data-testid="stSidebar"] {{ background-color: {CARD_BG}; }}
      [data-testid="stMetricValue"] {{ color: {TEXT}; }}
      [data-testid="stMetricLabel"] {{ color: {NEUTRAL}; }}
      h1, h2, h3, h4, h5 {{ color: {TEXT}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading customer scores…")
def load_customers() -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """Read the precomputed dashboard table.

    One CSV, four columns, no modelling at runtime. The customerID alignment
    that used to happen here now happens once in the build script, which asserts
    it before writing; by the time the file exists the four columns are already
    row-aligned by construction.
    """
    df = pd.read_csv(DATA_PATH)

    expected = {"customerID", "p_oof", "MonthlyCharges", "Contract"}
    missing = expected - set(df.columns)
    assert not missing, f"{DATA_PATH.name} is missing columns: {sorted(missing)}"
    assert df.isna().sum().sum() == 0, f"{DATA_PATH.name} contains NaNs"

    return (
        df["p_oof"].to_numpy(),
        df["MonthlyCharges"].to_numpy(),
        df["Contract"],
        df["customerID"],
    )


st.title("Retention Budget Optimizer")
st.caption(
    "Who to contact with a fixed retention budget, and what it is worth. "
    "Every figure is in euros under the assumptions set in the sidebar."
)

p, m, contract, ids = load_customers()

# Sliders are created once, outside the tabs: Streamlit widgets cannot be
# instantiated twice with the same identity, and both views are meant to answer
# the same scenario anyway.
params = sidebar_controls()

flat_tab, channel_tab = st.tabs(["Flat cost", "Cost by channel"])

with flat_tab:
    st.caption(
        "Every contact costs the same. The budget buys a fixed number of slots, "
        "so ranking by risk × value is already the best you can do — the "
        "knapsack would land on the same customers and is left off this tab."
    )
    render_decision_view("flat", p, m, contract, ids, params)

with channel_tab:
    st.caption(
        "Reaching a month-to-month customer needs an expensive channel; a "
        "two-year contract can be handled by a cheap automated touch. Once "
        "contact costs differ, picking who to contact becomes a packing "
        "problem and the optimiser stops agreeing with the ranking."
    )
    render_decision_view("channel", p, m, contract, ids, params)
