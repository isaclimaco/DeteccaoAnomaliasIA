import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from classifier_simple import SimpleMLP


transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder("datasets/classification/train", transform)
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)


model = SimpleMLP()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Treinando classificador simples...")

for epoch in range(10):
    total_loss = 0
    for imgs, labels in train_loader:
        out = model(imgs)
        loss = criterion(out, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/10 - Loss: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "models/simple_classifier.pth")
print("Modelo salvo em models/simple_classifier.pth")

