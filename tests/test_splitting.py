# tests/test_splitting.py
import math

import pandas as pd
import pytest

from forecastlab.splitting import chronological_split


@pytest.mark.parametrize("ratio", [0.1, 0.2, 0.25, 0.3, 0.4])
@pytest.mark.parametrize("n", [30, 77, 100])
def test_split_is_chronological_and_complete(ratio, n):
    s = pd.Series(range(n), index=pd.date_range("2024-01-01", periods=n))
    train, test = chronological_split(s, ratio)
    assert train.index.max() < test.index.min()
    assert len(test) == math.ceil(round(ratio * n, 9))
    assert len(train) + len(test) == n
    assert pd.concat([train, test]).equals(s)


@pytest.mark.parametrize("ratio", [0.0, 0.05, 0.5, 1.0])
def test_ratio_out_of_range_raises(ratio):
    s = pd.Series(range(30), index=pd.date_range("2024-01-01", periods=30))
    with pytest.raises(ValueError):
        chronological_split(s, ratio)