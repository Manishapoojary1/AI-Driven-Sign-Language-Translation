# train_multiclass.py  (FINAL VERSION)
import os
import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets, models
from tqdm import tqdm

DATA_DIR = "data/web_signs"
SAVE_DIR = "checkpoints"
EPOCHS = 7
BATCH_SIZE = 32
LR = 1e-4
IMG_SIZE = 224

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# transforms
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ColorJitter(0.3,0.3,0.3,0.1),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomPerspective(0.2),
    transforms.ToTensor(),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# dataset
dataset = datasets.ImageFolder(DATA_DIR, transform=train_tf)
classes = dataset.classes
num_classes = len(classes)

print("\nDetected classes:")
print(classes)

# split train/val
val_size = int(0.15 * len(dataset))
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
val_ds.dataset.transform = val_tf

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# model - ResNet18
model = models.resnet18(pretrained=True)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_acc = 0
os.makedirs(SAVE_DIR, exist_ok=True)

for epoch in range(1, EPOCHS+1):
    model.train()
    train_loss = 0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch} - Loss: {avg_loss:.4f}")

    # validation
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    print(f"Validation Accuracy: {acc:.2f}%")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_multiclass.pth"))
        print("✅ Saved Better Model")

# save metadata
with open(os.path.join(SAVE_DIR, "meta_multiclass.json"), "w") as f:
    json.dump({"classes": classes}, f, indent=2)

print("\n🎉 Training Complete!")
print("Best Accuracy:", best_acc)
