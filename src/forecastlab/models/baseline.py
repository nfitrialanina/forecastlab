# src/forecastlab/models/baseline.py
"""Model baseline: Naive dan Moving Average (prediksi satu langkah ke depan)."""
from __future__ import annotations

import pandas as pd


def naive_forecast(series: pd.Series, test_index: pd.Index) -> pd.Series:
    """Prediksi untuk waktu t adalah nilai aktual pada t-1."""
    return series.shift(1).loc[test_index].rename("naive")


def moving_average_forecast(series: pd.Series, test_index: pd.Index, window: int) -> pd.Series:
    """Prediksi untuk waktu t adalah rata-rata `window` nilai aktual sebelum t."""
    if window < 2:
        raise ValueError("window minimal 2.")
    return series.shift(1).rolling(window=window).mean().loc[test_index].rename("moving_average")
