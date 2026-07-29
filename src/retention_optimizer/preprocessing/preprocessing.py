"""End-to-end preprocessing: cleaning followed by feature engineering.

This is the single entry point the rest of the project should call to turn the
raw Telco dataframe into a model-ready one.
"""

import pandas as pd

from .pipelines.cleaning import clean_data
from .pipelines.feature_engineering import engineer_features


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full preprocessing pipeline: clean, then engineer features."""
    return df.pipe(clean_data).pipe(engineer_features)
