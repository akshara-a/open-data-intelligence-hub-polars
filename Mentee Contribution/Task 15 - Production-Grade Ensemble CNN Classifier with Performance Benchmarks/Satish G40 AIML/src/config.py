from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

IMG_HEIGHT = 32
IMG_WIDTH = 32
CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)
NUM_CLASSES = 10
BATCH_SIZE = 64
EPOCHS = 30
VALIDATION_SPLIT = 0.1
SEED = 42

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

MODEL_PATHS = {
    "baseline_cnn": MODELS_DIR / "baseline_cnn.keras",
    "regularized_cnn": MODELS_DIR / "regularized_cnn.keras",
    "deep_cnn": MODELS_DIR / "deep_cnn.keras",
}

for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR, NOTEBOOKS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
