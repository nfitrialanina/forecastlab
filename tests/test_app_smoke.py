# tests/test_app_smoke.py
"""Uji asap antarmuka: aplikasi dapat dimuat dan alur utama berjalan tanpa exception."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_app_loads_and_runs_forecast():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert "result" in at.session_state
    assert len(at.session_state["result"].metrics) == 3


def test_app_shows_error_when_date_and_target_are_the_same():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.selectbox[2].select("date")  # kolom target disamakan dengan kolom tanggal
    at.sidebar.button[0].click().run()
    assert not at.exception  # galat ditampilkan sebagai pesan, bukan stack trace
    assert [i.code for i in at.session_state["run_errors"]] == ["E-12"]
    assert "result" not in at.session_state
    