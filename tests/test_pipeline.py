# tests/test_pipeline.py
import io

import pandas as pd
import pytest

from forecastlab.data_loader import SAMPLE_DATASETS, load_sample
from forecastlab.errors import ForecastLabError
from forecastlab.export import params_to_json, results_to_csv
from forecastlab.pipeline import ForecastParams, run_forecast


@pytest.mark.parametrize("label", list(SAMPLE_DATASETS))
def test_end_to_end_on_samples(label):
    result = run_forecast(load_sample(label), "date", "value")
    assert result.train.index.max() < result.test.index.min()
    assert list(result.predictions.index) == list(result.test.index)
    assert set(result.predictions.columns) == {"naive", "moving_average", "random_forest"}
    assert len(result.metrics) == 3
    assert result.metrics[["mae", "rmse"]].notna().all().all()
    assert not result.predictions.isna().any().any()


def test_intermittent_sample_warns_about_mape():
    result = run_forecast(load_sample("Permintaan terputus-putus (sintetis)"), "date", "value")
    assert "W-03" in [w.code for w in result.warnings]


def test_pipeline_is_deterministic():
    df = load_sample("Tren + musiman (sintetis)")
    a = run_forecast(df, "date", "value")
    b = run_forecast(df, "date", "value")
    pd.testing.assert_frame_equal(a.predictions, b.predictions)


def test_e11_lag_too_large():
    df = load_sample("Tren + musiman (sintetis)")
    with pytest.raises(ForecastLabError) as e:
        run_forecast(df, "date", "value", ForecastParams(n_lags=570))
    assert e.value.code == "E-11"


def test_validation_errors_are_raised_as_forecastlab_error(make_df):
    df = make_df(40)
    with pytest.raises(ForecastLabError) as e:
        run_forecast(df, "date", "date")
    assert e.value.code == "E-12"


def test_short_test_set_warns(make_df):
    result = run_forecast(make_df(60), "date", "value", ForecastParams(test_ratio=0.1))
    assert "W-04" in [w.code for w in result.warnings]


def test_export_csv_roundtrip():
    result = run_forecast(load_sample("Tren + musiman (sintetis)"), "date", "value")
    back = pd.read_csv(io.BytesIO(results_to_csv(result)), parse_dates=["date"])
    assert list(back.columns) == [
        "date", "actual", "pred_naive", "pred_moving_average", "pred_random_forest",
    ]
    assert len(back) == len(result.test)
    assert back["actual"].tolist() == pytest.approx(result.test.tolist())


def test_export_params_json_is_valid():
    import json

    result = run_forecast(load_sample("Tren + musiman (sintetis)"), "date", "value")
    info = json.loads(params_to_json(result))
    assert info["n_train"] + info["n_test"] == info["n_total"]