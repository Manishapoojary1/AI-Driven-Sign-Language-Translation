import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import json

# --- 1. Basic Settings ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

data_dir = "data/asl_alphabet_train"
save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "best_asl_camera.pth")

# --- 2. Model Definition ---
class ASLModel(nn.Module):
    def __init__(self, num_classes=29):
        super(ASLModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 56 * 56, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# --- 3. Camera-style Augmentations ---
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
    transforms.RandomRotation(25),
    transforms.RandomHorizontalFlip(),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ToTensor(),
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# --- 4. Load Dataset ---
dataset = datasets.ImageFolder(root=data_dir, transform=train_tf)
val_dataset = datasets.ImageFolder(root=data_dir, transform=val_tf)

train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

num_classes = len(dataset.classes)
print(f"Detected {num_classes} classes: {dataset.classes}")

# --- 5. Initialize Model ---
model = ASLModel(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- 6. Train Loop ---
best_acc = 0
epochs = 3

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch"):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")

    # --- Validation ---
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = 100 * correct / total
    print(f"Validation Accuracy: {acc:.2f}%")

    # Save best model
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), save_path)
        print(f"✅ Saved improved model at {save_path}")

# --- 7. Save Labels Meta ---
meta_path = os.path.join(save_dir, "meta_camera.json")
with open(meta_path, "w") as f:
    json.dump(dataset.classes, f)
print(f"Saved label metadata to {meta_path}")
