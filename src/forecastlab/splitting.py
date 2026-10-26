# src/forecastlab/splitting.py
"""Pembagian data latih dan uji secara kronologis."""
from __future__ import annotations

import math

import pandas as pd

from . import config


def chronological_split(series: pd.Series, test_ratio: float) -> tuple[pd.Series, pd.Series]:
    """Bagi deret: data uji adalah ceil(p * n) observasi terakhir."""
    if not (config.TEST_RATIO_MIN <= test_ratio <= config.TEST_RATIO_MAX):
        raise ValueError(
            f"test_ratio harus di antara {config.TEST_RATIO_MIN} dan {config.TEST_RATIO_MAX}."
        )
    n_test = math.ceil(round(test_ratio * len(series), 9))  # round: hindari galat floating point
    if n_test >= len(series):
        raise ValueError("Data terlalu pendek untuk dibagi.")
    return series.iloc[:-n_test], series.iloc[-n_test:]