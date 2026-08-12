"""Build the single CSV the dashboard reads at runtime.

The dashboard needs four columns: the churn score, the monthly margin, the
contract type (for the per-channel cost policy) and the customer id. Three of
them live in `oof_predictions.csv`; `Contract` does not, because `encode()`
turns it into dummies before the OOF are written.

Rebuilding `Contract` means running `preprocess()` over the raw Telco file,
which drags scikit-learn, xgboost and joblib in. Doing it here, once, keeps all
of that out of the deployed app: the container only ever reads the CSV this
script produces.

Run from the repository root:
    uv run python scripts/build_dashboard_data.py
"""

from pathlib import Path

import pandas as pd

from retention_optimizer.evaluation.sweep import load_oof
from retention_optimizer.models.churn import load_data

ROOT = Path(__file__).resolve().parents[1]
OOF_PATH = ROOT / "data" / "processed" / "oof_predictions.csv"
RAW_PATH = ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUT_PATH = ROOT / "data" / "processed" / "dashboard_data.csv"

EXPECTED_ROWS = 7043
EXPECTED_CONTRACTS = {"Month-to-month", "One year", "Two year"}


def build() -> pd.DataFrame:
    """Join the OOF scores with `Contract`, aligned by customerID."""
    p, m, ids = load_oof(str(OOF_PATH))

    # Alignment by identifier, never by row order: the OOF csv and the S1 frame
    # come from different code paths and nothing guarantees they agree. Same
    # guard the notebook and the dashboard have always used.
    aligned = load_data(str(RAW_PATH)).set_index("customerID").reindex(ids)
    assert len(aligned) == len(ids), f"length mismatch: {len(aligned)} vs {len(ids)}"
    assert aligned.index.equals(pd.Index(ids)), "customerID order does not match"
    assert not aligned["Contract"].isna().any(), "customerIDs without a Contract"

    return pd.DataFrame(
        {
            "customerID": ids.values,
            "p_oof": p,
            "MonthlyCharges": m,
            "Contract": aligned["Contract"].values,
        }
    )


def validate(df: pd.DataFrame) -> None:
    """Fail loudly rather than ship a subtly broken file."""
    assert len(df) == EXPECTED_ROWS, f"expected {EXPECTED_ROWS} rows, got {len(df)}"
    assert df["customerID"].is_unique, "duplicate customerIDs"
    assert df.isna().sum().sum() == 0, "NaNs present"
    assert df["p_oof"].between(0, 1).all(), "p_oof outside [0, 1]"
    assert (df["MonthlyCharges"] > 0).all(), "non-positive MonthlyCharges"

    found = set(df["Contract"].unique())
    assert found == EXPECTED_CONTRACTS, f"unexpected Contract values: {found}"


if __name__ == "__main__":
    frame = build()
    validate(frame)
    frame.to_csv(OUT_PATH, index=False)

    print(f"written: {OUT_PATH.relative_to(ROOT)}")
    print(f"  rows    : {len(frame):,}")
    print(f"  columns : {list(frame.columns)}")
    print(f"  NaNs    : {int(frame.isna().sum().sum())}")
    print("  Contract:")
    for value, count in frame["Contract"].value_counts().items():
        print(f"    {value:<16} {count:5,}")
    print(f"  size    : {OUT_PATH.stat().st_size / 1024:.1f} KB")
