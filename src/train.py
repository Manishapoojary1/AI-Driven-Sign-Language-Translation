# src/train.py
import torch, os, time
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch import optim
import numpy as np
from pathlib import Path
import argparse
from sklearn.model_selection import train_test_split
import json

from dataset import KeypointDataset
from models import SeqClassifier
from config import DEVICE, CHECKPOINT_DIR, SEQUENCE_LEN

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

def collate_fn(batch):
    # batch: list of (kp_tensor (T,D), label)
    xs = torch.stack([b[0] for b in batch], dim=0)  # (B, T, D)
    ys = torch.stack([b[1] for b in batch], dim=0)
    return xs, ys

def train(manifest, epochs=30, batch_size=32, lr=1e-3, val_split=0.2, seed=42):
    ds = KeypointDataset(manifest)
    n = len(ds)
    idx = np.arange(n)
    # stratified split
    train_idx, val_idx = train_test_split(idx, test_size=val_split, random_state=seed, stratify=ds.df['label'])
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # get sample dims
    sample_kp, _ = ds[0]
    input_dim = sample_kp.shape[1]

    num_classes = len(ds.label_map)
    model = SeqClassifier(input_dim=input_dim, num_classes=num_classes).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val = 0.0
    meta = {"label_map": ds.label_map, "input_dim": input_dim}

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        start = time.time()
        for X, y in train_loader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            logits = model(X)
            loss = criterion(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += float(loss.item()) * y.size(0)
            preds = logits.argmax(dim=1)
            running_correct += (preds == y).sum().item()
            running_total += y.size(0)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total
        dur = time.time() - start

        # validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for Xv, yv in val_loader:
                Xv = Xv.to(DEVICE); yv = yv.to(DEVICE)
                logits_v = model(Xv)
                preds_v = logits_v.argmax(dim=1)
                val_correct += (preds_v == yv).sum().item()
                val_total += yv.size(0)
        val_acc = val_correct / max(1, val_total)

        print(f"Epoch {epoch}/{epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f} time={dur:.1f}s")

        # save best
        if val_acc > best_val:
            best_val = val_acc
            save_path = CHECKPOINT_DIR / "best.pth"
            torch.save({"model_state": model.state_dict(), "meta": meta}, save_path)
            print(f"Saved new best -> {save_path} (val_acc={val_acc:.4f})")

        # periodic save
        if epoch % 10 == 0:
            torch.save({"model_state": model.state_dict(), "meta": meta}, CHECKPOINT_DIR / f"epoch_{epoch}.pth")

    # final save metadata
    with open(CHECKPOINT_DIR / "meta.json", "w") as f:
        json.dump(meta, f)
    print("Training finished. Best val acc:", best_val)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default="data/labels.csv")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(args.manifest, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
