import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io

MODEL_PATH = "models/simple_classifier.pth"

# === MODELO SIMPLES MLP (MESMO DO TREINO) ===
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(3 * 64 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# === CARREGAMENTO ===
def load_classifier():
    model = SimpleMLP()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

model = load_classifier()

# === TRANSFORM ===
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

def classify_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_t = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, dim=1)

    label = "diseased" if pred.item() == 0 else "healthy"


    return label, confidence.item()
