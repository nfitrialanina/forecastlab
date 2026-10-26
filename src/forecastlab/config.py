# src/forecastlab/config.py
"""Seluruh konstanta dan nilai awal ForecastLab dipusatkan di sini."""

# Batas berkas
MAX_FILE_MB = 5
MAX_ROWS = 50_000

# Batas kualitas data
MIN_OBSERVATIONS = 30
MAX_MISSING_RATIO = 0.05
MAX_DUPLICATE_RATIO = 0.05
FREQ_TOLERANCE = 0.05

# Batas parameter
MIN_TRAIN_ROWS_AFTER_LAGS = 20
MIN_TEST_ROWS_WARNING = 10
TEST_RATIO_MIN = 0.1
TEST_RATIO_MAX = 0.4

# Nilai awal parameter
DEFAULT_TEST_RATIO = 0.2
DEFAULT_MA_WINDOW = 7
DEFAULT_N_LAGS = 7
DEFAULT_N_TREES = 200
RANDOM_SEED = 42

# MAPE: |aktual| <= epsilon dianggap nol
MAPE_EPSILON = 1e-8

MODEL_LABELS = {
    "naive": "Naive",
    "moving_average": "Moving Average",
    "random_forest": "Random Forest",
}