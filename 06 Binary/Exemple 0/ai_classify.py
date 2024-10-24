#!/usr/bin/env python3

import json
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# Imatges de test amb les seves etiquetes
test_images = [
    ["./data/test/img14469279.jpg", "non_cat"],
    ["./data/test/img15019810.jpg", "non_cat"],
    ["./data/test/img16615685.jpg", "non_cat"],
    ["./data/test/img16745259.jpg", "cat"],
    ["./data/test/img17242442.jpg", "cat"],
    ["./data/test/img21960791.jpg", "non_cat"],
    ["./data/test/img22921893.jpg", "cat"],
    ["./data/test/img23001964.jpg", "non_cat"],
    ["./data/test/img27753996.jpg", "non_cat"],
    ["./data/test/img30802655.jpg", "cat"],
    ["./data/test/img32929134.jpg", "non_cat"],
    ["./data/test/img34040492.jpg", "cat"],
    ["./data/test/img37438645.jpg", "non_cat"],
    ["./data/test/img38446080.jpg", "cat"],
    ["./data/test/img43753560.jpg", "non_cat"],
    ["./data/test/img44113566.jpg", "cat"],
    ["./data/test/img46733274.jpg", "non_cat"],
    ["./data/test/img47486374.jpg", "cat"],
    ["./data/test/img48140375.jpg", "cat"],
    ["./data/test/img49165968.jpg", "cat"],
    ["./data/test/img50470376.jpg", "cat"],
    ["./data/test/img53355576.jpg", "cat"],
    ["./data/test/img55000620.jpg", "cat"],
    ["./data/test/img57107487.jpg", "cat"],
    ["./data/test/img58115239.jpg", "non_cat"],
    ["./data/test/img62846124.jpg", "cat"],
    ["./data/test/img63161136.jpg", "non_cat"],
    ["./data/test/img69539582.jpg", "cat"],
    ["./data/test/img69679487.jpg", "non_cat"],
    ["./data/test/img69957115.jpg", "non_cat"],
    ["./data/test/img69968821.jpg", "non_cat"],
    ["./data/test/img70610683.jpg", "non_cat"],
    ["./data/test/img70610683.jpg", "non_cat"],
    ["./data/test/img72202194.jpg", "non_cat"],
    ["./data/test/img75381857.jpg", "non_cat"],
    ["./data/test/img75918332.jpg", "cat"],
    ["./data/test/img76888003.jpg", "cat"],
    ["./data/test/img77688616.jpg", "non_cat"],
    ["./data/test/img79053052.jpg", "cat"],
    ["./data/test/img83842359.jpg", "cat"],
    ["./data/test/img83918667.jpg", "cat"],
    ["./data/test/img84146180.jpg", "non_cat"],
    ["./data/test/img90037107.jpg", "cat"],
    ["./data/test/img93578086.jpg", "cat"],
    ["./data/test/img95378073.jpg", "non_cat"],
    ["./data/test/img95996327.jpg", "non_cat"],
    ["./data/test/img96295260.jpg", "non_cat"],
    ["./data/test/img96872108.jpg", "cat"],
    ["./data/test/img99363609.jpg", "non_cat"]
]

# Carregar configuració
with open('iscat_config.json', 'r') as f:
    config = json.load(f)

def create_model(num_classes):
    """Crear el mateix model que fem servir per entrenar"""
    model = resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(config['model_params']['dropout_rate']),
        nn.Linear(num_ftrs, config['model_params']['num_output']),
        nn.Sigmoid()
    )
    return model

def get_transform():
    """Obtenir les mateixes transformacions que fem servir per validació"""
    return transforms.Compose([
        transforms.Resize(tuple(config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['normalize_mean'], 
                           std=config['normalize_std'])
    ])

def load_model(model_path, num_classes, device):
    """Carregar el model amb els pesos entrenats"""
    model = create_model(num_classes)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    
    # Si hem guardat tot el checkpoint
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:  # Si només hem guardat els weights
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    return model

def evaluate_model(model, test_images, transform, class_names, device):
    """Avaluar el model en el conjunt de test"""
    correct = 0
    total = 0
    predictions = []
    
    with torch.no_grad():
        for img_path, true_label in tqdm(test_images, disable=True):
            # Carregar i preprocessar imatge
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            # Fer predicció
            outputs = model(img_tensor)
            predicted_prob = outputs.item()
            predicted_label = class_names[1] if predicted_prob > 0.5 else class_names[0]
            confidence = predicted_prob if predicted_prob > 0.5 else (1 - predicted_prob)
            
            # Guardar resultats
            is_correct = predicted_label == true_label
            if is_correct:
                correct += 1
            total += 1
            
            predictions.append({
                'image': Path(img_path).name,
                'predicted': predicted_label,
                'true_label': true_label,
                'confidence': confidence,
                'correct': is_correct
            })
            
            # Mostrar resultat
            print(f"Image: {Path(img_path).name}, Prediction: {confidence:.2%} = {"'"+predicted_label+"'":9} ({"'"+true_label+"'":9} > {'correct' if is_correct else 'wrong'})")    
            
    return correct, total

def main():
    # Configurar dispositiu
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         "cuda" if torch.cuda.is_available() else "cpu")
    if device == "mps" or device == "cuda":
        print(f"Using: '{device}' (GPU accelerated)")
    else:
        print(f"Using: '{device}' (not accelerated)")
    
    # Carregar classes
    class_names = config['classes']
    print(f"Classes: {class_names}")
    
    # Preparar model i transformacions
    model = load_model(config['model_path'], len(class_names), device)
    transform = get_transform()
    
    # Avaluar model
    correct, total = evaluate_model(
        model, test_images, transform, class_names, device
    )
    
    # Calcular i mostrar mètriques
    accuracy = correct / total
    print("\nGlobal results:")
    print(f"Total images: {total}")
    print(f"Hits: {correct}")
    print(f"Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()