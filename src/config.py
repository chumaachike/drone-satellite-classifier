from pathlib import Path
import os
import torch

#project root: drone-satellite-classifier/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

#check if code is working on Google Colab
IN_COLAB = "COLAB_GPU" in os.environ

#Base directories
if IN_COLAB:
    DRIVE_ROOT = Path("/content/drive/MyDrive/drone-satellite-classifier")
    DATA_DIR = DRIVE_DRIVE/"data"
    OUTPUT_DIR = DRIVE_ROOT / "outputs"
else:
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = PROJECT-ROOT / "output"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = OUTPUT_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = OUTPUT_DIR / "models"
PLOT_DIR = OUTPUT_DIR / "plots"
RESULT_DIR = OUTPUT_DIR / "results"

# Training settings
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
NUM_WORKERS = 2

# Reproducibility
SEED = 42

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Create output folders automatically
for path in [MODEL_DIR, PLOTS_DIR, RESULTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

