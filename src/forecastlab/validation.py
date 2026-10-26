# src/forecastlab/validation.py
"""Validasi kolom tanggal dan target. Tidak mengubah data."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from . import config
from .errors import Issue


@dataclass
class ValidationResult:
    errors: list[Issue] = field(default_factory=list)
    warnings: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_columns(df: pd.DataFrame, date_col: str, target_col: str) -> ValidationResult:
    """Periksa kolom tanggal dan target: E-05, E-06, E-07, E-08, E-12."""
    result = ValidationResult()

    for col in (date_col, target_col):
        if col not in df.columns:
            result.errors.append(Issue("E-05", f"Kolom '{col}' tidak ditemukan."))
    if result.errors:
        return result

    if date_col == target_col:
        result.errors.append(Issue("E-12", "Kolom tanggal dan kolom target tidak boleh sama."))
        return result

    # Kolom tanggal
    raw_date = df[date_col]
    if is_numeric_dtype(raw_date):
        result.errors.append(
            Issue("E-06", f"Kolom '{date_col}' berisi angka, bukan tanggal. Gunakan format YYYY-MM-DD.")
        )
    else:
        parsed = pd.to_datetime(raw_date, errors="coerce")
        bad = parsed.isna()
        if bad.any():
            examples = [str(v) for v in raw_date[bad].head(3).tolist()]
            result.errors.append(
                Issue(
                    "E-06",
                    f"{int(bad.sum())} nilai pada kolom '{date_col}' kosong atau tidak terbaca "
                    f"sebagai tanggal (contoh: {examples}). Gunakan format YYYY-MM-DD.",
                )
            )

    # Kolom target
    raw_target = df[target_col]
    numeric = pd.to_numeric(raw_target, errors="coerce")
    nonnumeric = raw_target.notna() & numeric.isna()
    if nonnumeric.any():
        examples = [str(v) for v in raw_target[nonnumeric].head(3).tolist()]
        result.errors.append(
            Issue(
                "E-07",
                f"Kolom '{target_col}' memuat {int(nonnumeric.sum())} nilai non-angka "
                f"(contoh: {examples}).",
            )
        )
    elif np.isinf(numeric.to_numpy(dtype=float)).any():
        result.errors.append(Issue("E-07", f"Kolom '{target_col}' memuat nilai tak hingga."))

    missing_ratio = float(numeric.isna().mean())
    if missing_ratio > config.MAX_MISSING_RATIO:
        result.errors.append(
            Issue(
                "E-08",
                f"Nilai hilang pada kolom '{target_col}' sebesar {missing_ratio:.1%}, "
                f"melebihi batas {config.MAX_MISSING_RATIO:.0%}.",
            )
        )
    return result