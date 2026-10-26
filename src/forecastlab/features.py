# src/forecastlab/features.py
"""Pembentukan fitur lag. Fitur untuk target t hanya memakai nilai sebelum t."""
from __future__ import annotations

import pandas as pd


def make_lag_features(series: pd.Series, n_lags: int) -> tuple[pd.DataFrame, pd.Series]:
    """Bentuk X (lag_1..lag_L) dan y. Baris awal dengan lag tidak lengkap dibuang."""
    if n_lags < 1:
        raise ValueError("n_lags minimal 1.")
    X = pd.DataFrame({f"lag_{k}": series.shift(k) for k in range(1, n_lags + 1)})
    mask = X.notna().all(axis=1)
    return X.loc[mask], series.loc[mask]