# tests/test_metrics.py
import math

import pytest

from forecastlab.metrics import compute_metrics, mae, mape, rmse


def test_mae_rmse_match_manual():
    y, yhat = [1, 2, 3], [2, 2, 5]  # galat: 1, 0, 2
    assert mae(y, yhat) == pytest.approx(1.0)
    assert rmse(y, yhat) == pytest.approx(math.sqrt(5 / 3))


def test_mape_matches_manual():
    value, excluded = mape([1, 2, 3], [2, 2, 5])
    assert value == pytest.approx(100 * (1 + 0 + 2 / 3) / 3)
    assert excluded == 0


def test_mape_excludes_zero_actuals():
    value, excluded = mape([0, 2, 4], [1, 1, 2])
    assert excluded == 1
    assert value == pytest.approx(50.0)  # hanya pada nilai aktual 2 dan 4


def test_mape_all_zero_is_nan_not_inf():
    value, excluded = mape([0, 0, 0], [1, 2, 3])
    assert math.isnan(value)
    assert excluded == 3


def test_mape_near_zero_is_excluded():
    value, excluded = mape([1e-9, 10.0], [1.0, 9.0])
    assert excluded == 1
    assert value == pytest.approx(10.0)


def test_compute_metrics_bundle():
    m = compute_metrics([0, 2, 4], [1, 1, 2])
    assert m.n_obs == 3 and m.n_mape_excluded == 1


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        mae([1, 2], [1])
        