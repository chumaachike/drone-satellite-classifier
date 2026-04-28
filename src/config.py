# src/config.py
from pathlib import Path
import os
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_COLAB = "COLAB_GPU" in os.environ

if IN_COLAB:
    DRIVE_ROOT = Path("/content/drive/MyDrive/drone-satellite-classifier")
    DATA_DIR = DRIVE_ROOT / "data"
    OUTPUT_DIR = DRIVE_ROOT / "outputs"
else:
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = PROJECT_ROOT / "outputs"

RAW_DATA_DIR = DATA_DIR / "raw"
DEGRADED_DATA_DIR = DATA_DIR / "degraded"

MODEL_DIR = OUTPUT_DIR / "models"
PLOT_DIR = OUTPUT_DIR / "plots"
RESULT_DIR = OUTPUT_DIR / "results"

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
NUM_WORKERS = 2
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

for path in [MODEL_DIR, PLOT_DIR, RESULT_DIR]:
    path.mkdir(parents=True, exist_ok=True)