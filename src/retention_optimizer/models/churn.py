"""Churn model: XGBoost training, calibration check and out-of-fold scoring.

Extracted verbatim from `notebooks/baseline.ipynb` (grid search) and
`notebooks/calibration_clv.ipynb` (calibration check, OOF generation). The
logic is copied as-is on purpose: any change to the split or the folds moves
the test set and invalidates the stored OOF probabilities in
`data/processed/oof_predictions.csv`, and everything downstream with them.

Cleaning, encoding and the canonical 80/20 split are deliberately NOT
reimplemented here — they already live in `retention_optimizer.preprocessing`
and that is what the notebooks call. A second copy of the split could drift
from the first one.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier

from retention_optimizer.preprocessing import preprocess

# Fixed across every notebook: the seed that pins the train/test partition and
# the CV folds. Do not change — the stored OOF depend on it.
RANDOM_STATE = 42
N_SPLITS = 5

# Grid from `baseline.ipynb`: tree complexity, shrinkage vs rounds, and
# row/feature sampling, scored on threshold-independent ROC-AUC.
PARAM_GRID = {
    "max_depth": [2, 3, 4],
    "min_child_weight": [1, 5],
    "learning_rate": [0.03, 0.1],
    "n_estimators": [200, 400],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}


def load_data(path: str) -> pd.DataFrame:
    """Read the raw Telco CSV and run the preprocessing pipeline.

    Cleaning (blank `TotalCharges` to 0, redundant "No * service" categories
    collapsed) and feature engineering happen inside `preprocess`.
    """
    return preprocess(pd.read_csv(path))


def train_xgb(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    """Grid-search the XGBoost classifier on the training set.

    Returns the refitted best estimator, scored by ROC-AUC over a 5-fold
    stratified CV.
    """
    search = GridSearchCV(
        estimator=XGBClassifier(
            tree_method="hist", eval_metric="logloss", random_state=RANDOM_STATE
        ),
        param_grid=PARAM_GRID,
        scoring="roc_auc",
        cv=StratifiedKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
        ),
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_


def load_xgb(path: str) -> XGBClassifier:
    """Rebuild an unfitted XGBoost from the hyperparameters of a saved model.

    Used to reuse the grid-searched configuration without re-running the
    search, exactly as `calibration_clv.ipynb` does.
    """
    return XGBClassifier(**joblib.load(path).get_params())


def calibration_scores(
    model: XGBClassifier, X_train: pd.DataFrame, y_train: pd.Series
) -> dict:
    """Compare the raw model against isotonic and sigmoid calibration.

    For each candidate, computes OOF probabilities on train and returns its
    Brier score plus the reliability-curve points, so the caller can plot
    them. The recorded decision was to keep the raw model: no wrapper
    improved on it.
    """
    estimators = {
        "XGB raw": model,
        "Isotonic": CalibratedClassifierCV(model, method="isotonic", cv=3),
        "Sigmoid": CalibratedClassifierCV(model, method="sigmoid", cv=3),
    }
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    results = {}
    for name, est in estimators.items():
        p = cross_val_predict(
            est, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
        )[:, 1]
        prob_true, prob_pred = calibration_curve(
            y_train, p, n_bins=10, strategy="quantile"
        )
        results[name] = {
            "brier": brier_score_loss(y_train, p),
            "prob_true": prob_true,
            "prob_pred": prob_pred,
        }
    return results


def compute_oof(
    model: XGBClassifier, X_full: pd.DataFrame, y_full: pd.Series
) -> np.ndarray:
    """Out-of-fold churn probabilities over the full dataset.

    No row is scored by a model that saw it during training. Per the
    calibration decision, the raw model is used directly with no
    `CalibratedClassifierCV` wrapper (5 fits instead of 15).
    """
    cv_full = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    return cross_val_predict(
        model, X_full, y_full, cv=cv_full, method="predict_proba", n_jobs=-1
    )[:, 1]


def build_oof_frame(
    customer_id: pd.Series, p_oof: np.ndarray, df_enc: pd.DataFrame
) -> pd.DataFrame:
    """Assemble the OOF table consumed by the optimization block.

    Columns: `customerID`, `p_oof` (calibrated churn probability) and
    `MonthlyCharges` (the margin proxy used as value).
    """
    oof_df = pd.DataFrame(
        {
            "customerID": customer_id.values,
            "p_oof": p_oof,
            "MonthlyCharges": df_enc["MonthlyCharges"].values,
        }
    )

    assert len(oof_df) == 7043
    assert oof_df["customerID"].is_unique
    assert oof_df["p_oof"].between(0, 1).all()
    assert oof_df.isna().sum().sum() == 0

    return oof_df
