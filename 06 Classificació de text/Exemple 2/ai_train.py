#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from ai_utils_text import load_config, save_metadata, save_model

CONFIG_FILE = "model_config.json"

def clearScreen():
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')

clearScreen()

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Training]")
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predicted = (outputs > 0.5).float()
        correct += (predicted == labels).float().sum().item()
        total += labels.numel()
        
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
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Validate]")
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).float().sum().item()
            total += labels.numel()
            
            pbar.set_postfix({
                'loss': f'{total_loss/len(val_loader):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    return total_loss/len(val_loader), 100.*correct/total

def check_early_stopping(state, val_loss):
    if state["best_loss"] is None or val_loss < state["best_loss"] - state["min_delta"]:
        state["best_loss"] = val_loss
        state["counter"] = 0
    else:
        state["counter"] += 1
        if state["counter"] >= state["patience"]:
            state["early_stop"] = True
    return state["early_stop"]

def configure_scheduler(optimizer, config):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=config['optimization']['scheduler']['reduce_lr_on_plateau']['mode'],
        factor=config['optimization']['scheduler']['reduce_lr_on_plateau']['factor'],
        patience=config['optimization']['scheduler']['reduce_lr_on_plateau']['patience']
    )

def main():
    config, unique_categories, model, device, train_loader, val_loader, vectorizer, label_encoder, early_stopping_state = load_config(CONFIG_FILE, mode="train", multi_label=True)
    
    criterion = nn.BCEWithLogitsLoss()  # Multi-label classification
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = configure_scheduler(optimizer, config)
    
    best_val_acc = 0.0
    
    for epoch in range(config['training']['epochs']):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch)
        
        print(f"Epoch {epoch+1}/{config['training']['epochs']} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, label_encoder, config['paths']['trained_network'])

        save_metadata(unique_categories, vectorizer, label_encoder, config['paths']['metadata'])   
   
        if check_early_stopping(early_stopping_state, val_loss):
            print("Early stopping activated")
            break
   
if __name__ == "__main__":
    main()
