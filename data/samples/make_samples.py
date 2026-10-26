# data/samples/make_samples.py
"""Membuat dua dataset contoh sintetis dengan seed tetap."""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
OUT_DIR = Path(__file__).parent


def make_trend_seasonal(n: int = 730, seed: int = SEED) -> pd.DataFrame:
    """Deret harian: tren naik + musiman mingguan + noise."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    t = np.arange(n)
    trend = 100 + 0.05 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 7)
    noise = rng.normal(0, 3, n)
    return pd.DataFrame({"date": dates, "value": (trend + seasonal + noise).round(2)})


def make_intermittent(n: int = 365, seed: int = SEED) -> pd.DataFrame:
    """Permintaan terputus-putus: sebagian besar hari bernilai nol."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    has_demand = rng.random(n) < 0.35
    sizes = rng.poisson(8, n) + 1
    values = np.where(has_demand, sizes, 0)
    return pd.DataFrame({"date": dates, "value": values})


def main() -> None:
    make_trend_seasonal().to_csv(OUT_DIR / "sample_trend_seasonal.csv", index=False)
    make_intermittent().to_csv(OUT_DIR / "sample_intermittent.csv", index=False)
    print("Dataset contoh dibuat di", OUT_DIR)


if __name__ == "__main__":
    main()