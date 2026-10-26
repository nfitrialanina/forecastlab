# app.py
"""ForecastLab — titik masuk aplikasi Streamlit. Berisi tata letak saja, bukan logika model."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from forecastlab import config
from forecastlab.data_loader import SAMPLE_DATASETS, load_csv_bytes, load_sample
from forecastlab.errors import ForecastLabError
from forecastlab.export import params_to_json, results_to_csv
from forecastlab.pipeline import ForecastParams, run_forecast
from forecastlab.preprocessing import prepare_series
from forecastlab.validation import validate_columns
from ui import charts, components

st.set_page_config(page_title="ForecastLab", page_icon="📈", layout="wide")


@st.cache_data(show_spinner=False)
def cached_sample(label: str) -> pd.DataFrame:
    return load_sample(label)  # hanya dataset contoh milik repositori yang di-cache


def load_data() -> tuple[str | None, pd.DataFrame | None]:
    source = st.sidebar.radio("Sumber data", ["Dataset contoh", "Unggah CSV"])
    if source == "Dataset contoh":
        label = st.sidebar.selectbox("Pilih dataset", list(SAMPLE_DATASETS))
        return label, cached_sample(label)
    upload = st.sidebar.file_uploader(
        f"Unggah berkas CSV (maks {config.MAX_FILE_MB} MB)", type=["csv"]
    )
    if upload is None:
        return None, None
    try:  # data unggahan hanya diproses di memori dan tidak di-cache
        return upload.name, load_csv_bytes(upload.getvalue(), upload.name)
    except ForecastLabError as err:
        components.show_issues(err.issues)
        return None, None


def main() -> None:
    st.title("📈 ForecastLab")
    st.caption("Platform interaktif untuk membandingkan model peramalan deret waktu.")

    name, df = load_data()
    if df is None:
        st.info("Pilih dataset contoh atau unggah CSV melalui sidebar untuk memulai.")
        return

    # Reset hasil lama bila data berganti.
    data_key = (name, df.shape)
    if st.session_state.get("data_key") != data_key:
        st.session_state["data_key"] = data_key
        st.session_state.pop("result", None)
        st.session_state.pop("run_errors", None)

    date_idx, target_idx = components.guess_columns(df)
    date_col = st.sidebar.selectbox("Kolom tanggal", list(df.columns), index=date_idx)
    target_col = st.sidebar.selectbox("Kolom target (numerik)", list(df.columns), index=target_idx)

    st.sidebar.subheader("Parameter")
    test_ratio = st.sidebar.slider(
        "Proporsi data uji", config.TEST_RATIO_MIN, config.TEST_RATIO_MAX,
        config.DEFAULT_TEST_RATIO, step=0.05,
    )
    ma_window = st.sidebar.number_input("Jendela Moving Average", 2, 60, config.DEFAULT_MA_WINDOW)
    n_lags = st.sidebar.number_input("Jumlah lag Random Forest", 1, 60, config.DEFAULT_N_LAGS)
    n_trees = st.sidebar.select_slider(
        "Jumlah pohon Random Forest", options=[50, 100, 200, 300, 500], value=config.DEFAULT_N_TREES
    )

    if st.sidebar.button("Jalankan peramalan", type="primary"):
        params = ForecastParams(test_ratio=float(test_ratio), ma_window=int(ma_window),
                                n_lags=int(n_lags), n_trees=int(n_trees))
        with st.spinner("Menjalankan model..."):
            try:
                st.session_state["result"] = run_forecast(df, date_col, target_col, params)
                st.session_state.pop("run_errors", None)
            except ForecastLabError as err:
                st.session_state["run_errors"] = err.issues
                st.session_state.pop("result", None)

    tab_data, tab_explore, tab_result, tab_download, tab_method = st.tabs(
        ["Data", "Eksplorasi", "Hasil", "Unduh", "Metodologi"]
    )

    with tab_data:
        st.subheader("Pratinjau data")
        st.dataframe(df.head(10), width="stretch")
        st.write(f"Ukuran data: **{len(df)} baris × {df.shape[1]} kolom**")
        check = validate_columns(df, date_col, target_col)
        if check.ok:
            st.success("Kolom terpilih lolos validasi.")
        else:
            components.show_issues(check.errors)

    with tab_explore:
        check = validate_columns(df, date_col, target_col)
        if not check.ok:
            st.info("Perbaiki pilihan kolom terlebih dahulu (lihat tab Data).")
        else:
            try:
                prepared = prepare_series(df, date_col, target_col)
            except ForecastLabError as err:
                components.show_issues(err.issues)
            else:
                components.show_issues(prepared.warnings, kind="warning")
                st.write(f"Frekuensi terdeteksi: **{prepared.freq}** · "
                         f"{len(prepared.series)} observasi setelah praproses")
                st.plotly_chart(charts.series_chart(prepared.series), width="stretch")
                st.dataframe(prepared.series.describe().to_frame("nilai"), width="stretch")

    with tab_result:
        if "run_errors" in st.session_state:
            components.show_issues(st.session_state["run_errors"])
        result = st.session_state.get("result")
        if result is None:
            st.info("Atur parameter lalu tekan **Jalankan peramalan** di sidebar.")
        else:
            components.show_issues(result.warnings, kind="warning")
            st.subheader("Perbandingan model")
            st.dataframe(components.format_metrics(result.metrics), hide_index=True,
                         width="stretch")
            st.caption("Tabel ini tidak memilih satu 'pemenang'. Peringkat dapat berbeda antar metrik "
                       "dan hanya berlaku untuk data dan pembagian ini.")
            st.plotly_chart(charts.forecast_chart(result), width="stretch")
            with st.expander("Diagnostik residual"):
                key = st.selectbox("Model", list(result.predictions.columns),
                                   format_func=lambda k: config.MODEL_LABELS[k])
                over_time, hist = charts.residual_charts(result, key)
                st.plotly_chart(over_time, width="stretch")
                st.plotly_chart(hist, width="stretch")

    with tab_download:
        result = st.session_state.get("result")
        if result is None:
            st.info("Belum ada hasil untuk diunduh.")
        else:
            st.download_button("Unduh prediksi (CSV)", results_to_csv(result),
                               file_name="forecastlab_hasil.csv", mime="text/csv")
            st.download_button("Unduh parameter (JSON)", params_to_json(result),
                               file_name="forecastlab_parameter.json", mime="application/json")

    with tab_method:
        st.markdown(METHODOLOGY)


METHODOLOGY = """
### Cara evaluasi
- **Pembagian kronologis:** data uji adalah bagian paling akhir dari deret; tidak ada pengacakan.
- **Satu langkah ke depan:** tiap titik uji diprediksi dari nilai aktual sebelumnya. Model dilatih sekali
  pada data latih dan tidak dilatih ulang selama periode uji.
- **Fitur lag:** target pada waktu *t* hanya memakai nilai sebelum *t*.
- **Nilai hilang:** diisi dengan nilai terakhir yang diketahui (bukan interpolasi), agar tidak memakai masa depan.

### Metrik
MAE dan RMSE dalam satuan data. **MAPE** membagi galat dengan nilai aktual sehingga tidak terdefinisi
saat aktual nol dan menyesatkan saat mendekati nol; observasi seperti itu dikecualikan dan jumlahnya ditampilkan.

### Keterbatasan
Hanya deret univariat berfrekuensi teratur; hasil berlaku untuk satu pembagian data dan tidak menjamin
kinerja di masa depan; tidak ada satu model yang selalu terbaik; data unggahan tidak disimpan.
"""

main()