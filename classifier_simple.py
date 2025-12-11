import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import os


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(64*64*3, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.classifier(x)



transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


def load_simple_model():
    model = SimpleMLP()
    if os.path.exists("models/simple_classifier.pth"):
        model.load_state_dict(torch.load("models/simple_classifier.pth", map_location="cpu"))
        model.eval()
    return model

model = load_simple_model()



CLASSES = ["healthy", "disease"]

def classify_simple(file_bytes):
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, 1)

    return {
        "class": CLASSES[pred.item()],
        "confidence": float(conf.item())
    }
