# src/dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path
from config import KEYPOINT_DIR

class KeypointDataset(Dataset):
    def __init__(self, manifest_csv: str, keypoint_dir: str = None, transform=None, label_map: dict = None):
        self.df = pd.read_csv(manifest_csv)
        self.keypoint_dir = Path(keypoint_dir) if keypoint_dir else Path(KEYPOINT_DIR)
        self.transform = transform
        # build or accept external label_map
        if label_map is None:
            labels = sorted(self.df['label'].unique())
            self.label_map = {lab: i for i, lab in enumerate(labels)}
        else:
            self.label_map = label_map
        self.inv_label_map = {v:k for k,v in self.label_map.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_id = str(row['id'])
        kp_file = self.keypoint_dir / f"{file_id}.npz"
        if not kp_file.exists():
            raise FileNotFoundError(f"Keypoint file missing: {kp_file}")
        data = np.load(kp_file)
        kp = data['kp'].astype(np.float32)  # (T, D)
        # optional per-sample normalization: subtract wrist position of first hand (landmark 0)
        # but keep as-is for now
        label_str = row['label']
        label = self.label_map[label_str]
        # return tensor (T, D) and label
        return torch.from_numpy(kp), torch.tensor(label, dtype=torch.long)
