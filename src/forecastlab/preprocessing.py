# src/forecastlab/preprocessing.py
"""Praproses kausal: urut, duplikat, frekuensi, dan pengisian nilai hilang."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from pandas.tseries.frequencies import to_offset

from . import config
from .errors import ForecastLabError, Issue


@dataclass
class PreparedSeries:
    series: pd.Series
    freq: str
    warnings: list[Issue] = field(default_factory=list)
    n_duplicates_removed: int = 0
    n_missing_filled: int = 0
    n_leading_dropped: int = 0


def _fail(code: str, message: str) -> ForecastLabError:
    return ForecastLabError(Issue(code, message))


def _regular_grid(index: pd.DatetimeIndex) -> tuple[pd.DatetimeIndex, str]:
    """Kembalikan grid tanggal teratur yang mencakup seluruh observasi, atau galat E-09."""
    freq = pd.infer_freq(index)
    if freq is not None:
        return pd.date_range(index[0], periods=len(index), freq=freq), freq

    step = index.to_series().diff().dropna().mode().iloc[0]
    grid = pd.date_range(index[0], index[-1], freq=to_offset(step))
    gap_ratio = (len(grid) - len(index)) / len(grid)
    if not index.isin(grid).all() or gap_ratio > config.FREQ_TOLERANCE:
        raise _fail(
            "E-09",
            f"Frekuensi data tidak teratur (selisih dominan: {step}). Pastikan data harian, "
            "mingguan, atau bulanan dengan celah tidak lebih dari "
            f"{config.FREQ_TOLERANCE:.0%}.",
        )
    return grid, str(to_offset(step).freqstr)


def prepare_series(df: pd.DataFrame, date_col: str, target_col: str) -> PreparedSeries:
    """Ubah DataFrame mentah menjadi Series bertanggal teratur tanpa nilai hilang."""
    warnings: list[Issue] = []
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col], errors="coerce"),
            "value": pd.to_numeric(df[target_col], errors="coerce"),
        }
    )
    data = data.dropna(subset=["date"]).sort_values("date", kind="stable")
    if data.empty:
        raise _fail("E-10", "Tidak ada data yang dapat diproses.")

    n_dup = int(data.duplicated(subset="date", keep="last").sum())
    if n_dup / len(data) > config.MAX_DUPLICATE_RATIO:
        raise _fail(
            "E-13",
            f"{n_dup} baris memiliki tanggal duplikat (melebihi {config.MAX_DUPLICATE_RATIO:.0%}). "
            "Agregasikan data per tanggal sebelum diunggah.",
        )
    data = data.drop_duplicates(subset="date", keep="last")
    if n_dup:
        warnings.append(
            Issue("W-01", f"{n_dup} baris duplikat tanggal dihapus (observasi terakhir dipakai).")
        )

    series = data.set_index("date")["value"]
    if len(series) < config.MIN_OBSERVATIONS:
        raise _fail(
            "E-10",
            f"Data hanya {len(series)} observasi; minimal {config.MIN_OBSERVATIONS} diperlukan.",
        )

    grid, freq = _regular_grid(series.index)
    series = series.reindex(grid)
    n_missing = int(series.isna().sum())
    series = series.ffill()  # kausal: hanya memakai nilai masa lalu
    n_leading = int(series.isna().sum())
    series = series.dropna()
    n_filled = n_missing - n_leading

    if len(series) < config.MIN_OBSERVATIONS:
        raise _fail(
            "E-10",
            f"Setelah praproses tersisa {len(series)} observasi; "
            f"minimal {config.MIN_OBSERVATIONS} diperlukan.",
        )
    if n_filled:
        warnings.append(
            Issue("W-02", f"{n_filled} nilai hilang diisi dengan nilai terakhir yang diketahui.")
        )

    series.name = target_col
    series.index.name = "date"
    return PreparedSeries(
        series=series,
        freq=freq,
        warnings=warnings,
        n_duplicates_removed=n_dup,
        n_missing_filled=n_filled,
        n_leading_dropped=n_leading,
    )