# tests/test_data_loader.py
import pytest

from forecastlab import config
from forecastlab.data_loader import SAMPLE_DATASETS, load_csv_bytes, load_sample
from forecastlab.errors import ForecastLabError

CSV = b"date,value\n2024-01-01,1\n2024-01-02,2\n"


def code_of(excinfo) -> str:
    return excinfo.value.code


def test_valid_csv_loads():
    df = load_csv_bytes(CSV, "data.csv")
    assert list(df.columns) == ["date", "value"]
    assert len(df) == 2


def test_semicolon_separator_is_detected():
    df = load_csv_bytes(b"date;value\n2024-01-01;1\n2024-01-02;2\n", "data.csv")
    assert list(df.columns) == ["date", "value"]


def test_e01_wrong_extension():
    with pytest.raises(ForecastLabError) as e:
        load_csv_bytes(CSV, "data.xlsx")
    assert code_of(e) == "E-01"


def test_e02_file_too_large(monkeypatch):
    monkeypatch.setattr(config, "MAX_FILE_MB", 0)
    with pytest.raises(ForecastLabError) as e:
        load_csv_bytes(CSV, "data.csv")
    assert code_of(e) == "E-02"


def test_e03_bad_encoding():
    with pytest.raises(ForecastLabError) as e:
        load_csv_bytes(b"date,value\n\xff\xfe\xfa,1\n", "data.csv")
    assert code_of(e) == "E-03"


def test_e04_too_many_rows(monkeypatch):
    monkeypatch.setattr(config, "MAX_ROWS", 1)
    with pytest.raises(ForecastLabError) as e:
        load_csv_bytes(CSV, "data.csv")
    assert code_of(e) == "E-04"


@pytest.mark.parametrize("content", [b"", b"   \n", b"only_one_column\n1\n2\n"])
def test_e05_empty_or_single_column(content):
    with pytest.raises(ForecastLabError) as e:
        load_csv_bytes(content, "data.csv")
    assert code_of(e) == "E-05"


@pytest.mark.parametrize("label", list(SAMPLE_DATASETS))
def test_samples_load(label):
    df = load_sample(label)
    assert list(df.columns) == ["date", "value"]