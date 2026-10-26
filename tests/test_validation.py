# tests/test_validation.py
import numpy as np
import pandas as pd

from forecastlab.validation import validate_columns


def codes(result) -> list[str]:
    return [i.code for i in result.errors]


def test_valid_data_passes(make_df):
    assert validate_columns(make_df(), "date", "value").ok


def test_e05_missing_column(make_df):
    assert "E-05" in codes(validate_columns(make_df(), "tanggal", "value"))


def test_e12_same_column(make_df):
    assert codes(validate_columns(make_df(), "date", "date")) == ["E-12"]


def test_e06_unparseable_date(make_df):
    df = make_df()
    df.loc[3, "date"] = "bukan tanggal"
    assert "E-06" in codes(validate_columns(df, "date", "value"))


def test_e06_numeric_date_column(make_df):
    df = make_df()
    df["date"] = np.arange(len(df))
    assert "E-06" in codes(validate_columns(df, "date", "value"))


def test_e07_text_in_target(make_df):
    df = make_df()
    df["value"] = df["value"].astype(object)
    df.loc[2, "value"] = "abc"
    assert "E-07" in codes(validate_columns(df, "date", "value"))


def test_e07_infinite_target(make_df):
    df = make_df()
    df.loc[2, "value"] = np.inf
    assert "E-07" in codes(validate_columns(df, "date", "value"))


def test_e08_too_many_missing(make_df):
    df = make_df(40)
    df.loc[:4, "value"] = np.nan  # 5 dari 40 = 12,5%
    assert "E-08" in codes(validate_columns(df, "date", "value"))


def test_small_missing_ratio_is_allowed(make_df):
    df = make_df(100)
    df.loc[2, "value"] = np.nan
    assert validate_columns(df, "date", "value").ok


def test_validation_does_not_modify_data(make_df):
    df = make_df()
    before = df.copy()
    validate_columns(df, "date", "value")
    pd.testing.assert_frame_equal(df, before)