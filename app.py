import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# 1) ALPHABET MODEL (A–Z)
# =====================================================
class ASLAlphabetModel(nn.Module):
    def __init__(self, num_classes=29):
        super().__init__()
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
        return self.fc(x)


alphabet_model_path = "checkpoints/best_asl.pth"
alphabet_labels = [chr(i) for i in range(65, 91)] + ["space", "del", "nothing"]

alphabet_model = None

if os.path.exists(alphabet_model_path):
    alphabet_model = ASLAlphabetModel(num_classes=len(alphabet_labels))
    alphabet_model.load_state_dict(torch.load(alphabet_model_path, map_location=DEVICE))
    alphabet_model.to(DEVICE)
    alphabet_model.eval()
    print("✅ Alphabet model loaded.")
else:
    print("⚠ Alphabet model NOT found!")


# =====================================================
# 2) SIGN-WORD MODEL (hello, bye, yes, no, thank_you)
# =====================================================
sign_model_path = "checkpoints/best_multiclass.pth"
sign_meta_path = "checkpoints/meta_multiclass.json"

sign_model = None
sign_labels = None

if os.path.exists(sign_model_path) and os.path.exists(sign_meta_path):
    with open(sign_meta_path, "r") as f:
        sign_labels = json.load(f)["classes"]

    sign_model = models.resnet18(weights=None)
    sign_model.fc = nn.Linear(sign_model.fc.in_features, len(sign_labels))
    sign_model.load_state_dict(torch.load(sign_model_path, map_location=DEVICE))
    sign_model.to(DEVICE)
    sign_model.eval()

    print("✅ Sign-word model loaded:", sign_labels)
else:
    print("⚠ Sign-word model NOT found!")


# =====================================================
# 3) Transform
# =====================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# =====================================================
# 4) Prediction alphabet
# =====================================================
def predict_alphabet(image):
    if alphabet_model is None:
        return "Alphabet model missing!"

    img = Image.fromarray(image)
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = alphabet_model(x)
        pred = torch.argmax(out).item()

    return alphabet_labels[pred]


# =====================================================
# 5) Prediction sign words
# =====================================================
def predict_sign_word(image):
    if sign_model is None:
        return "Sign-word model missing! Train it first."

    img = Image.fromarray(image)
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():        # FIXED HERE
        out = sign_model(x)
        pred = torch.argmax(out).item()

    return sign_labels[pred]


# =====================================================
# 6) UI
# =====================================================
alphabet_tab = gr.Interface(
    fn=predict_alphabet,
    inputs=gr.Image(type="numpy", label="Upload Alphabet Sign Image"),
    outputs="text",
    title="Alphabet Detection (A–Z)"
)

sign_tab = gr.Interface(
    fn=predict_sign_word,
    inputs=gr.Image(type="numpy", label="Upload Sign Word Image"),
    outputs="text",
    title="Sign Word Detection",
    description="Upload hello / bye / thank_you / yes / no images ONLY from your dataset."
)

ui = gr.TabbedInterface([alphabet_tab, sign_tab],
                        ["🔤 Alphabet", "🤟 Sign Words"])

ui.launch()
