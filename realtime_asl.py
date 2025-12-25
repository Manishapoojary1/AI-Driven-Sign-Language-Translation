import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import json

# --- 1. Model Definition ---
class ASLModel(nn.Module):
    def __init__(self, num_classes=29):  # 29 to match trained model
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 56 * 56, 256), nn.ReLU(), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# --- 2. Load model and labels ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "checkpoints/best_asl.pth"
meta_path = "checkpoints/meta.json"

with open(meta_path, "r") as f:
    meta = json.load(f)

labels = meta.get("classes", [chr(i) for i in range(65, 91)] + ["space", "del", "nothing"])

model = ASLModel(num_classes=len(labels))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# --- 3. Start webcam ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not detected")
    exit()

print("✅ Starting real-time ASL detection... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # define region of interest (ROI)
    x1, y1, x2, y2 = 100, 100, 324, 324
    roi = frame[y1:y2, x1:x2]

    # draw rectangle around ROI
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # preprocess ROI
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    from PIL import Image
    img = Image.fromarray(img)
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img)
        pred = torch.argmax(out, dim=1).item()

    label = labels[pred]
    cv2.putText(frame, f"Prediction: {label}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("ASL Real-Time Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
