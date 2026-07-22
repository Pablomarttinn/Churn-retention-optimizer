"""Preprocessing package: cleaning, feature engineering, encoding and split."""

from .pipelines.encoding import encode, scale_numeric, split_data
from .preprocessing import preprocess

__all__ = ["preprocess", "encode", "split_data", "scale_numeric"]
