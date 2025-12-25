# train_asl.py  – minimal trainer that works with your app.py
from pathlib import Path
import torch, torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import json, os

# --- same architecture as in app.py ---
class ASLModel(nn.Module):
    def __init__(self, num_classes=26):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(32*56*56,256), nn.ReLU(), nn.Linear(256,num_classes)
        )
    def forward(self,x):
        x=self.conv(x)
        x=x.view(x.size(0),-1)
        return self.fc(x)

# --- simple dataset loader (expects data/train/<A-Z>/images.jpg) ---
def train_model(data_dir='data/asl_alphabet_train', epochs=3, batch_size=32, lr=1e-3):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    ds = datasets.ImageFolder(root=data_dir, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    print("Found", len(ds), "images and", len(ds.classes), "classes:", ds.classes)

    model = ASLModel(num_classes=len(ds.classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    for ep in range(1, epochs+1):
        model.train()
        running = 0
        for imgs, lbls in tqdm(loader, desc=f"Epoch {ep}"):
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = model(imgs)
            loss = crit(out, lbls)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item()
        print(f"Epoch {ep} loss={running/len(loader):.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/best_asl.pth")
    json.dump({"classes": ds.classes}, open("checkpoints/meta.json","w"))
    print("✅ Model saved to checkpoints/best_asl.pth")

if __name__=="__main__":
    train_model()
