# train_asl.py -- uses existing data/asl_alphabet_train folder
import os, json, torch, torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

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

def train_model(data_dir='data/asl_alphabet_train', epochs=3, batch_size=32, lr=1e-3):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using', device)
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    ds = datasets.ImageFolder(root=data_dir, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    print(f'Loaded {len(ds)} images across {len(ds.classes)} classes.')
    model = ASLModel(num_classes=len(ds.classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(1, epochs+1):
        model.train()
        total_loss=0
        for imgs, labels in tqdm(loader, desc=f'Epoch {ep}'):
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            out = model(imgs)
            loss = loss_fn(out, labels)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f'Epoch {ep}: loss={total_loss/len(loader):.4f}')
    Path('checkpoints').mkdir(exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/best_asl.pth')
    json.dump({'classes': ds.classes}, open('checkpoints/meta.json','w'))
    print('✅ Saved checkpoints/best_asl.pth')

if __name__ == '__main__':
    train_model()
