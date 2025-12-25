# src/infer.py
import torch, numpy as np
from pathlib import Path
from models import SeqModel
from config import DEVICE
import argparse

def load_model(ckpt_path, sample_input_dim):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    label_map = ckpt['label_map']
    num_classes = len(label_map)
    model = SeqModel(input_dim=sample_input_dim, num_classes=num_classes).to(DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    inv_label_map = {v:k for k,v in label_map.items()}
    return model, inv_label_map

def predict(npz_path, ckpt_path):
    data = np.load(npz_path)
    kp = data['kp'].astype('float32')
    sample_dim = kp.shape[1]
    model, inv = load_model(ckpt_path, sample_dim)
    x = torch.from_numpy(kp[None]).to(DEVICE)  # (1, T, D)
    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(1).item()
    return inv[pred]

if __name__ == "__main__":
    import fire
    fire.Fire(predict)
