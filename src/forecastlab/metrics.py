# src/forecastlab/metrics.py
"""Metrik evaluasi: MAE, RMSE, dan MAPE dengan penanganan nilai nol."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config


@dataclass(frozen=True)
class MetricResult:
    mae: float
    rmse: float
    mape: float  # NaN jika seluruh observasi dikecualikan
    n_mape_excluded: int
    n_obs: int


def _as_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("y_true dan y_pred harus berukuran sama dan tidak kosong.")
    if np.isnan(a).any() or np.isnan(b).any():
        raise ValueError("y_true dan y_pred tidak boleh memuat NaN.")
    return a, b


def mae(y_true, y_pred) -> float:
    a, b = _as_arrays(y_true, y_pred)
    return float(np.mean(np.abs(a - b)))


def rmse(y_true, y_pred) -> float:
    a, b = _as_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mape(y_true, y_pred, epsilon: float | None = None) -> tuple[float, int]:
    """MAPE (%) hanya pada observasi dengan |aktual| > epsilon.

    Mengembalikan (nilai, jumlah observasi dikecualikan). Nilai NaN bila semuanya dikecualikan.
    """
    eps = config.MAPE_EPSILON if epsilon is None else epsilon
    a, b = _as_arrays(y_true, y_pred)
    keep = np.abs(a) > eps
    n_excluded = int((~keep).sum())
    if not keep.any():
        return float("nan"), n_excluded
    value = float(np.mean(np.abs((a[keep] - b[keep]) / a[keep])) * 100)
    return value, n_excluded


def compute_metrics(y_true, y_pred) -> MetricResult:
    mape_value, n_excluded = mape(y_true, y_pred)
    return MetricResult(
        mae=mae(y_true, y_pred),
        rmse=rmse(y_true, y_pred),
        mape=mape_value,
        n_mape_excluded=n_excluded,
        n_obs=int(np.asarray(y_true).size),
    )