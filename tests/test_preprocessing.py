import numpy as np
import pandas as pd
import pytest

from forecastlab.errors import ForecastLabError
from forecastlab.preprocessing import prepare_series


def warning_codes(prepared) -> list[str]:
    return [w.code for w in prepared.warnings]


def test_clean_daily_data(make_df):
    p = prepare_series(make_df(40), "date", "value")
    assert len(p.series) == 40
    assert p.freq == "D"
    assert p.warnings == []


def test_unsorted_input_is_sorted(make_df):
    df = make_df(40).sample(frac=1, random_state=0)
    p = prepare_series(df, "date", "value")
    assert p.series.index.is_monotonic_increasing


def test_missing_value_is_filled_causally(make_df):
    df = make_df(40)
    df.loc[10, "value"] = np.nan
    p = prepare_series(df, "date", "value")
    assert p.series.iloc[10] == 9.0  # nilai sebelumnya, BUKAN rata-rata dengan nilai sesudahnya
    assert "W-02" in warning_codes(p)
    assert p.n_missing_filled == 1


def test_leading_missing_is_dropped(make_df):
    df = make_df(40)
    df.loc[0, "value"] = np.nan
    p = prepare_series(df, "date", "value")
    assert len(p.series) == 39
    assert p.series.iloc[0] == 1.0
    assert p.n_leading_dropped == 1


def test_duplicate_dates_keep_last(make_df):
    df = make_df(40)
    dup = pd.DataFrame({"date": [df.loc[5, "date"]], "value": [99.0]})
    p = prepare_series(pd.concat([df, dup], ignore_index=True), "date", "value")
    assert p.series.loc[df.loc[5, "date"]] == 99.0
    assert "W-01" in warning_codes(p)
    assert len(p.series) == 40


def test_e13_too_many_duplicates(make_df):
    df = make_df(40)
    df = pd.concat([df, df.iloc[:10]], ignore_index=True)
    with pytest.raises(ForecastLabError) as e:
        prepare_series(df, "date", "value")
    assert e.value.code == "E-13"


def test_small_calendar_gap_is_filled(make_df):
    df = make_df(40).drop(index=10).reset_index(drop=True)
    p = prepare_series(df, "date", "value")
    assert len(p.series) == 40
    assert p.series.iloc[10] == 9.0
    assert "W-02" in warning_codes(p)


def test_e09_irregular_frequency():
    gaps = [1, 3] * 20
    dates = pd.Timestamp("2024-01-01") + pd.to_timedelta(np.cumsum(gaps), unit="D")
    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "value": np.arange(40.0)})
    with pytest.raises(ForecastLabError) as e:
        prepare_series(df, "date", "value")
    assert e.value.code == "E-09"


def test_e10_too_short(make_df):
    with pytest.raises(ForecastLabError) as e:
        prepare_series(make_df(20), "date", "value")
    assert e.value.code == "E-10"


def test_weekly_frequency_is_detected(make_df):
    p = prepare_series(make_df(40, freq="W"), "date", "value")
    assert p.freq.startswith("W")
    assert len(p.series) == 40