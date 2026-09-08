from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
DATA_FILE = DATA_DIR / "Customer Complaints Sentiment and Priority Dataset.csv"
SENTIMENT_MODEL_FILE = MODEL_DIR / "sentiment_model.joblib"
PRODUCT_MODEL_FILE = MODEL_DIR / "product_model.joblib"
PRODUCT_ENCODER_FILE = MODEL_DIR / "product_encoder.joblib"
PRIORITY_MODEL_FILE = MODEL_DIR / "priority_model.joblib"
TRAINING_TEXTS_FILE = MODEL_DIR / "training_texts.joblib"
EVALUATION_FILE = MODEL_DIR / "evaluation.txt"
RANDOM_STATE = 42
TEST_SIZE = 0.20
