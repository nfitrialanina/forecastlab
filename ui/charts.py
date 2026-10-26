# ui/charts.py
"""Grafik Plotly. Hanya menerima data; tidak berisi logika model."""
from __future__ import annotations

import plotly.graph_objects as go

from forecastlab import config
from forecastlab.pipeline import ForecastResult


def series_chart(series) -> go.Figure:
    fig = go.Figure(go.Scatter(x=series.index, y=series.values, mode="lines", name=str(series.name)))
    fig.update_layout(
        title="Deret waktu", xaxis_title="Tanggal", yaxis_title=str(series.name),
        margin={"l": 10, "r": 10, "t": 50, "b": 10}, hovermode="x unified",
    )
    return fig


def forecast_chart(result: ForecastResult, show_last_train: int = 100) -> go.Figure:
    """Aktual vs prediksi. Hanya `show_last_train` titik latih terakhir ditampilkan agar terbaca."""
    train_tail = result.train.iloc[-show_last_train:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_tail.index, y=train_tail.values, mode="lines",
                             name="Data latih", line={"color": "gray"}))
    fig.add_trace(go.Scatter(x=result.test.index, y=result.test.values, mode="lines",
                             name="Aktual (uji)", line={"color": "black", "width": 3}))
    for key in result.predictions.columns:
        fig.add_trace(go.Scatter(x=result.predictions.index, y=result.predictions[key],
                                 mode="lines", name=config.MODEL_LABELS[key],
                                 line={"dash": "dot"}))
    start = result.test.index[0].to_pydatetime()
    fig.add_shape(type="line", x0=start, x1=start, y0=0, y1=1, yref="paper",
                  line={"color": "red", "dash": "dash"})
    fig.add_annotation(x=start, y=1, yref="paper", text="Awal data uji", showarrow=False,
                       yanchor="bottom")
    fig.update_layout(title="Aktual vs prediksi pada data uji", xaxis_title="Tanggal",
                      yaxis_title=str(result.series.name), hovermode="x unified",
                      margin={"l": 10, "r": 10, "t": 60, "b": 10})
    return fig


def residual_charts(result: ForecastResult, model_key: str) -> tuple[go.Figure, go.Figure]:
    """Residual = aktual - prediksi: grafik terhadap waktu dan histogram."""
    residual = result.test - result.predictions[model_key]
    label = config.MODEL_LABELS[model_key]
    over_time = go.Figure(go.Scatter(x=residual.index, y=residual.values, mode="lines+markers"))
    over_time.add_hline(y=0, line_dash="dash")
    over_time.update_layout(title=f"Residual terhadap waktu ({label})",
                            margin={"l": 10, "r": 10, "t": 50, "b": 10})
    hist = go.Figure(go.Histogram(x=residual.values))
    hist.update_layout(title=f"Distribusi residual ({label})", margin={"l": 10, "r": 10, "t": 50, "b": 10})
    return over_time, hist
