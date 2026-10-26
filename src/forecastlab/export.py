# src/forecastlab/export.py
"""Ekspor hasil ke CSV dan JSON dalam bentuk bytes (tidak menulis ke disk)."""
from __future__ import annotations

import json
from dataclasses import asdict

import pandas as pd

from .pipeline import ForecastResult


def results_to_csv(result: ForecastResult) -> bytes:
    """CSV: date, actual, pred_<model> untuk periode uji."""
    df = pd.DataFrame({"date": result.test.index, "actual": result.test.to_numpy()})
    for key in result.predictions.columns:
        df[f"pred_{key}"] = result.predictions[key].to_numpy()
    return df.to_csv(index=False).encode("utf-8")


def params_to_json(result: ForecastResult) -> bytes:
    """Ringkasan parameter dan ukuran data untuk reproduksi."""
    info = {
        "params": asdict(result.params),
        "freq": result.freq,
        "n_total": len(result.series),
        "n_train": len(result.train),
        "n_test": len(result.test),
        "test_start": str(result.test.index[0].date()),
        "test_end": str(result.test.index[-1].date()),
        "warnings": [f"{w.code}: {w.message}" for w in result.warnings],
    }
    return json.dumps(info, indent=2, ensure_ascii=False).encode("utf-8")
