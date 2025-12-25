# src/config.py
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
KEYPOINT_DIR = DATA_DIR / "keypoints"
KEYPOINT_DIR.mkdir(parents=True, exist_ok=True)

# keypoint settings
MAX_NUM_HANDS = 2
NUM_LANDMARKS = 21  # mediapipe hand landmarks per hand
KP_DIM = 3  # x,y,z

# sequence and training defaults
SEQUENCE_LEN = 64    # frames per clip (pad or trim)
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
