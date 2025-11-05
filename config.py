# src/config.py
from pathlib import Path

# --- Project Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
DOCS_DIR = BASE_DIR / "docs"

# --- Image-to-Image (Sketch to Photo) Settings ---
IMG2IMG_RAW_DIR = RAW_DATA_DIR / "sketch_to_photo"
IMG2IMG_PROCESSED_DIR = PROCESSED_DATA_DIR / "sketch_to_photo"

# --- Crowd Density Settings ---
CROWD_RAW_DIR = RAW_DATA_DIR / "crowd_density"
CROWD_PROCESSED_DIR = PROCESSED_DATA_DIR / "crowd_density"

# --- Preprocessing & Augmentation Parameters ---
# Most Image-to-Image models (like Pix2Pix) use 256x256
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3

# Normalization for GANs is often to [-1, 1]
# (pixel_value / 127.5) - 1.0  => results in [-1, 1]
NORMALIZATION_MODE = "gan" # "gan" or "standard"

# Augmentation settings
AUGMENTATION_COUNT = 5 # How many augmented versions to create per image