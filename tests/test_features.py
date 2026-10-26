# tests/test_features.py
import pandas as pd
import pytest

from forecastlab.features import make_lag_features


def test_shape_and_first_row(daily_series):
    X, y = make_lag_features(daily_series, 3)
    assert list(X.columns) == ["lag_1", "lag_2", "lag_3"]
    assert len(X) == len(daily_series) - 3  # 3 baris awal dibuang
    first = X.iloc[0]
    assert (first["lag_1"], first["lag_2"], first["lag_3"]) == (2.0, 1.0, 0.0)
    assert y.iloc[0] == 3.0


def test_no_future_information(daily_series):
    """TC-04: mengubah nilai pada waktu t dan sesudahnya tidak boleh mengubah fitur sampai t."""
    t = 8
    changed = daily_series.copy()
    changed.iloc[t:] += 1000
    X1, _ = make_lag_features(daily_series, 3)
    X2, _ = make_lag_features(changed, 3)
    upto = daily_series.index[t]
    pd.testing.assert_frame_equal(X1.loc[:upto], X2.loc[:upto])


def test_invalid_lag_raises(daily_series):
    with pytest.raises(ValueError):
        make_lag_features(daily_series, 0)