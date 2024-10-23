#!/usr/bin/env python3

# python3 -m pip install torch torchvision torchaudio Pillow tqdm --break-system-package

import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

model_path = 'vehicle_classifier.pth'
types_path = 'vehicle_types.json'

# Imatges de test amb les seves etiquetes
test_images = [
    ['./data/test/571d8914dd5d1b7e5a493b167ee49ce6.jpg', 'family_sedan'],
    ['./data/test/b00473a9f517bbfc5923c95288b40ff2.jpg', 'taxi'],
    ['./data/test/ffe56e1e84047ba47c423fc9937924c1.jpg', 'truck'],
    ['./data/test/5789a84b08a72667f6ed1dc0e1400778.jpg', 'jeep'],
    ['./data/test/b0322b716be09e89ac21c22874194836.jpg', 'heavy_truck']
]

print("Define transformations")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Carregar les etiquetes des de l'arxiu JSON
with open(types_path, 'r') as f:
    class_names = json.load(f)

num_classes = len(class_names)  # Nombre de classes basat en el JSON

print("Define AI model")
model = resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)  # Ajusta a la quantitat de classes

# Carregar els pesos del model entrenat
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
if device.type == "cuda" or device.type == "mps":
    print(f"Using device: {device} (GPU accelerated)")
else:
    print(f"Using device: {device} (CPU based)")

print("Start classification")
predictions = {}

with torch.no_grad():
    for img_path, label in test_images:
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_label = predicted.item()

        predictions[img_path] = {
            'predicted_class': class_names[predicted_label],  # Mapeja la predicció a la classe
            'true_class': label
        }

# Mostrar les prediccions per pantalla
for img_path, result in predictions.items():
    print(f"Image: {img_path}, Predicted Class: {result['predicted_class']}, True Class: {result['true_class']}")

print("Classification completed.")
