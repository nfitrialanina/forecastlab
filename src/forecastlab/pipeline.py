# src/forecastlab/pipeline.py
"""Orkestrasi seluruh alur: validasi -> praproses -> split -> model -> metrik."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import config
from .errors import ForecastLabError, Issue
from .features import make_lag_features
from .metrics import compute_metrics
from .models.baseline import moving_average_forecast, naive_forecast
from .models.random_forest import random_forest_forecast
from .preprocessing import prepare_series
from .splitting import chronological_split
from .validation import validate_columns


@dataclass
class ForecastParams:
    test_ratio: float = config.DEFAULT_TEST_RATIO
    ma_window: int = config.DEFAULT_MA_WINDOW
    n_lags: int = config.DEFAULT_N_LAGS
    n_trees: int = config.DEFAULT_N_TREES
    seed: int = config.RANDOM_SEED


@dataclass
class ForecastResult:
    series: pd.Series
    train: pd.Series
    test: pd.Series
    predictions: pd.DataFrame  # indeks = tanggal uji, kolom = kunci model
    metrics: pd.DataFrame  # satu baris per model
    warnings: list[Issue] = field(default_factory=list)
    params: ForecastParams = field(default_factory=ForecastParams)
    freq: str = ""


def _check_params(params: ForecastParams, n_train: int) -> None:
    problems: list[Issue] = []
    if not 2 <= params.ma_window <= n_train:
        problems.append(
            Issue("E-11", f"Jendela Moving Average harus antara 2 dan {n_train} (data latih).")
        )
    if params.n_lags < 1 or n_train - params.n_lags < config.MIN_TRAIN_ROWS_AFTER_LAGS:
        problems.append(
            Issue(
                "E-11",
                f"Jumlah lag terlalu besar untuk {n_train} data latih: sisa baris latih harus "
                f"minimal {config.MIN_TRAIN_ROWS_AFTER_LAGS}. Kecilkan lag atau pakai data lebih panjang.",
            )
        )
    if problems:
        raise ForecastLabError(problems)


def run_forecast(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    params: ForecastParams | None = None,
) -> ForecastResult:
    """Jalankan alur lengkap. Melempar ForecastLabError bila masukan tidak layak."""
    params = params or ForecastParams()

    validation = validate_columns(df, date_col, target_col)
    if not validation.ok:
        raise ForecastLabError(validation.errors)

    prepared = prepare_series(df, date_col, target_col)
    series = prepared.series
    warnings = list(validation.warnings) + list(prepared.warnings)

    train, test = chronological_split(series, params.test_ratio)
    _check_params(params, n_train=len(train))
    if len(test) < config.MIN_TEST_ROWS_WARNING:
        warnings.append(
            Issue("W-04", f"Data uji hanya {len(test)} observasi; metrik bisa tidak stabil.")
        )

    # Fitur lag dibentuk dari seluruh deret, tetapi tiap baris hanya memuat nilai SEBELUM targetnya.
    X_all, y_all = make_lag_features(series, params.n_lags)
    train_rows = X_all.index.intersection(train.index)
    X_train, y_train = X_all.loc[train_rows], y_all.loc[train_rows]
    X_test = X_all.loc[test.index]

    predictions = pd.DataFrame(
        {
            "naive": naive_forecast(series, test.index),
            "moving_average": moving_average_forecast(series, test.index, params.ma_window),
            "random_forest": random_forest_forecast(
                X_train, y_train, X_test, params.n_trees, params.seed
            ),
        }
    )

    rows = []
    for key in predictions.columns:
        m = compute_metrics(test.to_numpy(), predictions[key].to_numpy())
        rows.append(
            {
                "model": key,
                "mae": m.mae,
                "rmse": m.rmse,
                "mape": m.mape,
                "n_mape_excluded": m.n_mape_excluded,
            }
        )
    metrics = pd.DataFrame(rows)

    n_excluded = int(metrics["n_mape_excluded"].iloc[0])
    if n_excluded == len(test):
        warnings.append(
            Issue("W-05", "Seluruh nilai aktual uji bernilai nol; MAPE tidak tersedia.")
        )
    elif n_excluded > 0:
        warnings.append(
            Issue(
                "W-03",
                f"{n_excluded} dari {len(test)} nilai aktual uji nol atau mendekati nol "
                "dan dikecualikan dari MAPE. Utamakan MAE dan RMSE.",
            )
        )

    return ForecastResult(
        series=series,
        train=train,
        test=test,
        predictions=predictions,
        metrics=metrics,
        warnings=warnings,
        params=params,
        freq=prepared.freq,
    )
