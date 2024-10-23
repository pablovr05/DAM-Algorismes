#!/usr/bin/env python3

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm

# Constants
MODEL_PATH = 'iscat_model.pth'
TYPES_PATH = 'iscat_classes.json'
BATCH_SIZE = 16
EPOCHS = 25
LEARNING_RATE = 0.0001
VALIDATION_SPLIT = 0.2

# Transformacions
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def initialize_early_stopping(patience=7, min_delta=0):
    return {
        "patience": patience,
        "min_delta": min_delta,
        "counter": 0,
        "best_loss": None,
        "early_stop": False
    }

def check_early_stopping(state, val_loss):
    if state["best_loss"] is None or val_loss < state["best_loss"] - state["min_delta"]:
        state["best_loss"] = val_loss
        state["counter"] = 0
    else:
        state["counter"] += 1
        if state["counter"] >= state["patience"]:
            state["early_stop"] = True
    return state["early_stop"]

def create_data_loaders():
    print("Carregant dataset...")
    full_dataset = datasets.ImageFolder('./data/train', transform=transform)
    
    if len(full_dataset.classes) != 2:
        raise ValueError("El dataset ha de tenir exactament 2 classes per classificació binària")
    
    val_size = int(VALIDATION_SPLIT * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"Dataset carregat: {train_size} imatges d'entrenament, {val_size} de validació")
    return train_loader, val_loader, full_dataset.classes

def create_model():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    
    # Congelar capes
    for param in list(model.parameters())[:-4]:
        param.requires_grad = False
        
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 1),  # Una única sortida per classificació binària
        nn.Sigmoid()  # Sigmoid per obtenir probabilitat entre 0 i 1
    )
    return model

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]")
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.float().to(device)  # Convertir a float per BCE
        
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predicted = (outputs > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({
            'loss': f'{total_loss/len(train_loader):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss/len(train_loader), 100.*correct/total

def validate(model, val_loader, criterion, device, epoch):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]")
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.float().to(device)
            
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f'{total_loss/len(val_loader):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    return total_loss/len(val_loader), 100.*correct/total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilitzant dispositiu: {device}")
    
    train_loader, val_loader, classes = create_data_loaders()
    print(f"Classes trobades: {classes}")
    
    with open(TYPES_PATH, 'w') as f:
        json.dump(classes, f)
    
    model = create_model()
    model = model.to(device)
    print(f"Model creat amb {sum(p.numel() for p in model.parameters() if p.requires_grad)} pesos i biaixos")
    
    criterion = nn.BCELoss()  # Binary Cross Entropy per classificació binària
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    early_stopping_state = initialize_early_stopping(patience=10)
    
    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch)
        
        print(f"""Epoch {epoch}/{EPOCHS} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - LR: {optimizer.param_groups[0]['lr']:.6f} """)
        
        scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"Guardant millor model amb accuracy {val_acc:.2f}%")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, MODEL_PATH)
        
        if check_early_stopping(early_stopping_state, val_loss):
            print("Early stopping activat")
            break

if __name__ == "__main__":
    main()