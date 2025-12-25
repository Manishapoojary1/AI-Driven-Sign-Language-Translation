# train_multiclass_asl_split.py
import os, json
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

DATA_TRAIN = Path("data/combined/images/train")
DATA_TEST  = Path("data/combined/images/test")
SAVE_DIR   = Path("checkpoints")
SAVE_DIR.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# transforms
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7,1.0)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomRotation(12),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
test_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# datasets
if not DATA_TRAIN.exists():
    raise SystemExit(f"Train folder not found: {DATA_TRAIN.resolve()}")
if not DATA_TEST.exists():
    raise SystemExit(f"Test folder not found: {DATA_TEST.resolve()}")

train_ds = datasets.ImageFolder(root=str(DATA_TRAIN), transform=train_tf)
test_ds  = datasets.ImageFolder(root=str(DATA_TEST),  transform=test_tf)

print(f"Train classes ({len(train_ds.classes)}): {train_ds.classes}")
print(f"Test classes  ({len(test_ds.classes)}): {test_ds.classes}")

if train_ds.classes != test_ds.classes:
    print("Warning: class lists differ between train and test. Please ensure same class folders exist in both.")
    
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

# model: use pretrained ResNet18 (better than tiny conv from scratch)
model = models.resnet18(pretrained=True)
num_feats = model.fc.in_features
model.fc = nn.Linear(num_feats, len(train_ds.classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, verbose=True)

best_acc = 0.0
best_path = SAVE_DIR / "best_multiclass_asl_split.pth"

EPOCHS = 6  # change to 3 for very fast test, 6+ for better accuracy

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} train")
    for imgs, labels in pbar:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    train_loss = running_loss / len(train_loader.dataset)
    
    # validate on test
    model.eval()
    correct = 0; total = 0; val_loss = 0.0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device); labels = labels.to(device)
            out = model(imgs)
            loss = criterion(out, labels)
            val_loss += loss.item() * imgs.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_loss = val_loss / len(test_loader.dataset)
    val_acc = 100.0 * correct / total
    print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.2f}%")
    
    scheduler.step(val_loss)
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({"model_state": model.state_dict(), "classes": train_ds.classes}, str(best_path))
        print("Saved best model:", best_path)

# save classes to meta.json
meta = {"classes": train_ds.classes}
with open(SAVE_DIR / "meta_multiclass_split.json", "w") as f:
    json.dump(meta, f)
print("Training finished. Best val acc:", best_acc)
print("Meta saved to:", SAVE_DIR / "meta_multiclass_split.json")
