# src/evaluate.py
import torch
from torch.utils.data import DataLoader
from dataset import KeypointDataset
from models import SeqModel
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from config import DEVICE
import joblib
from pathlib import Path

def evaluate(manifest, ckpt_path):
    ds = KeypointDataset(manifest)
    dl = DataLoader(ds, batch_size=64)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    label_map = ckpt['label_map']
    num_classes = len(label_map)
    sample_kp, _ = ds[0]
    model = SeqModel(input_dim=sample_kp.shape[1], num_classes=num_classes).to(DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for X, y in dl:
            X = X.to(DEVICE)
            logits = model(X)
            preds = logits.argmax(1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(y.numpy().tolist())
    print(classification_report(y_true, y_pred, target_names=[k for k,_ in sorted(label_map.items(), key=lambda x:x[1])]))
    cm = confusion_matrix(y_true, y_pred)
    np.save("confusion_matrix.npy", cm)

if __name__ == "__main__":
    import fire
    fire.Fire(evaluate)
