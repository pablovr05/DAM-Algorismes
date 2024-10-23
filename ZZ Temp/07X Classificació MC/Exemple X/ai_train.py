#!/usr/bin/env python3

# python3 -m pip install torch torchvision torchaudio Pillow tqdm --break-system-package

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm

epochs = 5
model_path = 'vehicle_classifier.pth'
types_path = 'vehicle_types.json'

print("Define transformations")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print("Load data set")
train_dataset = datasets.ImageFolder('./data/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
types = train_dataset.classes
print("Types:", types)
with open(types_path, 'w') as f:
    json.dump(types, f)

print("Define AI model")
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))

print("Set up optimization")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
if device.type == "cuda" or device.type == "mps":
    print(f"Using device: {device} (GPU accelerated)")
else:
    print(f"Using device: {device} (CPU based)")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    # Utilitzar tqdm per mostrar el progrés de cada lot dins d'un epoch
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} Progress", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch}, Average Loss: {running_loss / len(train_loader):.4f}")

print(f"Save trained model at {model_path}")
torch.save(model.state_dict(), model_path)
