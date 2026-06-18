import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_RAW = BASE_DIR / "data" / "hour.csv"
DATA_PROCESSED = BASE_DIR / "data" / "processed.csv"
MODELS_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "plots"

MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "best_model.joblib"
TUNED_MODEL_PATH = MODELS_DIR / "tuned_model.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.2

TARGET = "cnt"

CATEGORICAL_COLS = ["season", "weathersit"]
NUMERIC_WEATHER_COLS = ["temp", "atemp", "hum", "windspeed"]
