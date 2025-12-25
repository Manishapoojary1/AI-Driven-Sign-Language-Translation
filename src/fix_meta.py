# src/fix_meta.py
import sys
from pathlib import Path

# add project root to sys.path so we can import src.dataset
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import json
from src.dataset import KeypointDataset
