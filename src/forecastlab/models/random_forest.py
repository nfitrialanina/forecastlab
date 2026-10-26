# src/forecastlab/models/random_forest.py
"""Random Forest dengan fitur lag."""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def random_forest_forecast(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    n_trees: int,
    seed: int,
) -> pd.Series:
    """Latih HANYA pada data latih, lalu prediksi baris uji."""
    model = RandomForestRegressor(n_estimators=n_trees, random_state=seed, n_jobs=-1)
    model.fit(X_train.to_numpy(), y_train.to_numpy())
    pred = model.predict(X_test.to_numpy())
    return pd.Series(pred, index=X_test.index, name="random_forest")
