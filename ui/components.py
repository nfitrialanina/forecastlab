# ui/components.py
"""Komponen Streamlit kecil yang dipakai app.py."""
from __future__ import annotations

import pandas as pd
import streamlit as st
from pandas.api.types import is_numeric_dtype

from forecastlab import config
from forecastlab.errors import Issue


def guess_columns(df: pd.DataFrame) -> tuple[int, int]:
    """Tebak indeks kolom tanggal dan target untuk nilai awal pilihan."""
    cols = list(df.columns)
    date_idx = 0
    for i, c in enumerate(cols):
        if is_numeric_dtype(df[c]):
            continue
        if pd.to_datetime(df[c].head(20), errors="coerce").notna().all():
            date_idx = i
            break
    target_idx = next(
        (i for i, c in enumerate(cols) if i != date_idx and is_numeric_dtype(df[c])),
        min(1, len(cols) - 1),
    )
    return date_idx, target_idx


def show_issues(issues: list[Issue], kind: str = "error") -> None:
    show = st.error if kind == "error" else st.warning
    for issue in issues:
        show(f"**{issue.code}** — {issue.message}")


def format_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Tabel untuk tampilan: label model, angka dibulatkan, MAPE NaN jadi 'tidak tersedia'."""
    out = pd.DataFrame(
        {
            "Model": metrics["model"].map(config.MODEL_LABELS),
            "MAE": metrics["mae"].round(4),
            "RMSE": metrics["rmse"].round(4),
            "MAPE (%)": metrics["mape"].map(lambda v: "tidak tersedia" if pd.isna(v) else f"{v:.2f}"),
            "Obs. dikecualikan dari MAPE": metrics["n_mape_excluded"],
        }
    )
    return out
