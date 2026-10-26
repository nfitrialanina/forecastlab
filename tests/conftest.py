# tests/conftest.py
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def make_df():
    """Pabrik DataFrame uji: kolom 'date' (teks ISO) dan 'value' (0, 1, 2, ...)."""

    def _make(n: int = 40, start: str = "2024-01-01", freq: str = "D") -> pd.DataFrame:
        dates = pd.date_range(start, periods=n, freq=freq).strftime("%Y-%m-%d")
        return pd.DataFrame({"date": dates, "value": np.arange(n, dtype=float)})

    return _make


@pytest.fixture
def daily_series() -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=12, freq="D")
    return pd.Series(np.arange(12, dtype=float), index=idx, name="value")