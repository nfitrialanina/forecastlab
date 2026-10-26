# src/forecastlab/data_loader.py
"""Membaca CSV (unggahan atau dataset contoh) ke DataFrame. Tidak menulis ke disk."""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from . import config
from .errors import ForecastLabError, Issue

SAMPLES_DIR = Path(__file__).resolve().parents[2] / "data" / "samples"

SAMPLE_DATASETS: dict[str, str] = {
    "Tren + musiman (sintetis)": "sample_trend_seasonal.csv",
    "Permintaan terputus-putus (sintetis)": "sample_intermittent.csv",
}


def _fail(code: str, message: str) -> ForecastLabError:
    return ForecastLabError(Issue(code, message))


def load_csv_bytes(data: bytes, filename: str) -> pd.DataFrame:
    """Baca isi berkas CSV (bytes) menjadi DataFrame setelah pemeriksaan dasar."""
    if not filename.lower().endswith(".csv"):
        raise _fail("E-01", "Berkas harus berekstensi .csv.")
    if len(data) > config.MAX_FILE_MB * 1024 * 1024:
        raise _fail("E-02", f"Ukuran berkas melebihi batas {config.MAX_FILE_MB} MB.")
    if not data.strip():
        raise _fail("E-05", "Berkas kosong.")

    try:
        df = pd.read_csv(io.BytesIO(data), sep=",", encoding="utf-8-sig")
        if df.shape[1] == 1:  # coba pemisah lain (titik koma atau tab)
            for sep in (";", "\t"):
                alt = pd.read_csv(io.BytesIO(data), sep=sep, encoding="utf-8-sig")
                if alt.shape[1] > 1:
                    df = alt
                    break
    except UnicodeDecodeError:
        raise _fail("E-03", "Berkas tidak dapat dibaca. Simpan ulang sebagai CSV UTF-8.") from None
    except pd.errors.EmptyDataError:
        raise _fail("E-05", "Berkas tidak berisi data.") from None
    except pd.errors.ParserError:
        raise _fail("E-03", "Format CSV tidak valid. Periksa pemisah kolom dan tanda kutip.") from None

    if len(df) > config.MAX_ROWS:
        raise _fail("E-04", f"Jumlah baris ({len(df)}) melebihi batas {config.MAX_ROWS}.")
    if df.shape[1] < 2 or df.empty:
        raise _fail("E-05", "Berkas harus memiliki minimal dua kolom dan satu baris data.")

    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_sample(label: str) -> pd.DataFrame:
    """Muat salah satu dataset contoh berdasarkan labelnya."""
    path = SAMPLES_DIR / SAMPLE_DATASETS[label]
    return load_csv_bytes(path.read_bytes(), path.name)