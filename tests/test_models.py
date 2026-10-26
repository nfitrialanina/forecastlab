# tests/test_models.py
import numpy as np
import pandas as pd
import pytest

from forecastlab.features import make_lag_features
from forecastlab.models.baseline import moving_average_forecast, naive_forecast
from forecastlab.models.random_forest import random_forest_forecast


@pytest.fixture
def small():
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    return pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=idx)


def test_naive_matches_manual(small):
    pred = naive_forecast(small, small.index[3:])
    assert pred.tolist() == [3.0, 4.0, 5.0]


def test_moving_average_matches_manual(small):
    pred = moving_average_forecast(small, small.index[3:], window=2)
    # t=3 -> (2+3)/2, t=4 -> (3+4)/2, t=5 -> (4+5)/2
    assert pred.tolist() == [2.5, 3.5, 4.5]


def test_baselines_do_not_use_current_value(small):
    changed = small.copy()
    changed.iloc[3] = 1000.0
    a = moving_average_forecast(small, small.index[3:4], window=2)
    b = moving_average_forecast(changed, changed.index[3:4], window=2)
    assert a.iloc[0] == b.iloc[0]


def test_moving_average_window_too_small(small):
    with pytest.raises(ValueError):
        moving_average_forecast(small, small.index[3:], window=1)


def test_random_forest_is_reproducible():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(size=80).cumsum(), index=pd.date_range("2024-01-01", periods=80))
    X, y = make_lag_features(s, 5)
    X_train, y_train, X_test = X.iloc[:50], y.iloc[:50], X.iloc[50:]
    a = random_forest_forecast(X_train, y_train, X_test, n_trees=30, seed=42)
    b = random_forest_forecast(X_train, y_train, X_test, n_trees=30, seed=42)
    pd.testing.assert_series_equal(a, b)
    assert list(a.index) == list(X_test.index)
    