#!/usr/bin/env python3

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm

# Configuració
config = {
    "config_path": "iscat_config.json",
    "model_path": "iscat_model.pth",
    "training": {
        "batch_size": 16,
        "epochs": 25,
        "learning_rate": 0.0001,
        "validation_split": 0.2
    },
    "image_size": [224, 224],
    "early_stopping": {
        "patience": 10,
        "min_delta": 0
    },
    "reduce_lr_on_plateau": {
        "mode": "min",
        "factor": 0.1,
        "patience": 5
    },
    "normalize_mean": [0.485, 0.456, 0.406],
    "normalize_std": [0.229, 0.224, 0.225],
    "model_params": {
        "dropout_rate": 0.5,
        "num_output": 1
    },
    "classes": []
}

EPOCHS = config['training']['epochs']

# Transformacions
# La variable transform defineix una sèrie de transformacions 
# que s'aplicaran a cada imatge abans d'introduir-la al model. 
# Aquestes transformacions preparen les dades de manera consistent 
# per optimitzar el rendiment de l'entrenament.
transform = transforms.Compose([
    # Redimensiona la imatge a la mida especificada al paràmetre 'image_size' de la configuració
    transforms.Resize(tuple(config['image_size'])),
    # Converteix la imatge a un tensor, canviant la representació de píxels a valors numèrics
    transforms.ToTensor(),
    # Normalitza els valors del tensor utilitzant la mitjana i la desviació estàndard especificades
    transforms.Normalize(mean=config['normalize_mean'], std=config['normalize_std'])
])

# La funció create_model crea una versió modificada de la xarxa ResNet18, 
# utilitzant pesos preentrenats a partir d'ImageNet.
# ResNet18 és un tipus de xarxa neuronal convolucional (CNN)
# ImageNet és una base de dades d'imatges usada per entrenar xarxes neurals
def create_model():
    # Crea una instància del model ResNet18 preentrenat amb pesos de la base de dades ImageNet
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    
    # Congelar les capes inicials per evitar que es modifiquin durant l'entrenament
    # Això ajuda a mantenir els pesos preentrenats en les capes primeres, que ja han après característiques útils
    for param in list(model.parameters())[:-4]:
        param.requires_grad = False
        
    # Obtenir el nombre de característiques d'entrada de la capa final de classificació
    num_ftrs = model.fc.in_features
    
    # Substituir la capa final per una nova seqüència que inclou:
    # - Una capa de Dropout per reduir el sobreajustament durant l'entrenament
    # - Una capa Lineal (fully connected) que transforma les característiques en el nombre de sortides desitjat
    # - Una activació Sigmoid per produir sortides entre 0 i 1, útil per a classificació binària o multietiquetes
    model.fc = nn.Sequential(
        nn.Dropout(config['model_params']['dropout_rate']),
        nn.Linear(num_ftrs, config['model_params']['num_output']),
        nn.Sigmoid()
    )
    
    # Retornar el model modificat
    return model

# La funció create_data_loaders carrega imatges des de la carpeta ./data/train 
# i les prepara per ser utilitzades durant l'entrenament i validació del model.
def create_data_loaders():
    print("Loading dataset...")
    
    # Càrrega del conjunt de dades des de la carpeta './data/train' aplicant les transformacions definides a 'transform'
    full_dataset = datasets.ImageFolder('./data/train', transform=transform)
    
    # Comprovar si el dataset té exactament 2 classes, ja que s'espera una classificació binària
    if len(full_dataset.classes) != 2:
        raise ValueError("Dataset must have 2 classes for binary classification")
    
    # Definir la mida del conjunt de validació segons un percentatge especificat a la configuració
    val_size = int(config['training']['validation_split'] * len(full_dataset))
    
    # La resta de les dades s'utilitzen per al conjunt d'entrenament
    train_size = len(full_dataset) - val_size
    
    # Dividir el conjunt complet en subconjunts d'entrenament i de validació
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Crear un DataLoader per al conjunt d'entrenament amb un tamany de batch definit i amb barrejament activat
    train_loader = DataLoader(train_dataset, 
                              batch_size=config['training']['batch_size'], 
                              shuffle=True, 
                              num_workers=2)
    
    # Crear un DataLoader per al conjunt de validació sense barrejament (per fer la validació consistent)
    val_loader = DataLoader(val_dataset, 
                            batch_size=config['training']['batch_size'], 
                            shuffle=False, 
                            num_workers=2)
    
    # Missatge de consola indicant quantes imatges hi ha als conjunts d'entrenament i validació
    print(f"Dataset loaded: {train_size} training images, {val_size}")
    
    # Retornar els DataLoaders creats juntament amb les classes del dataset
    return train_loader, val_loader, full_dataset.classes

# Crea i inicialitza un diccionari amb els paràmetres i l'estat necessaris 
# per implementar la tècnica d'early stopping durant l'entrenament del model. 
# Aquesta tècnica permet aturar l'entrenament de manera prematura si 
# el model deixa de millorar, evitant el sobreajustament
def initialize_early_stopping():
    # Retornar un diccionari que conté els paràmetres i estat per a la lògica d'early stopping
    return {
        # Nombre d'epochs consecutius sense millora permesos abans d'aturar l'entrenament
        "patience": config['early_stopping']['patience'],

        # La mínima millora en pèrdua necessària per considerar que hi ha hagut una millora
        "min_delta": config['early_stopping']['min_delta'],
        
        # Comptador d'epochs sense millora
        "counter": 0,
        
        # Millor valor de pèrdua registrat fins ara
        "best_loss": None,
        
        # Indicador per saber si s'ha d'aturar l'entrenament prematurament
        "early_stop": False
    }

# Comprova si s'ha complert la condició per aturar l'entrenament 
# prematurament mitjançant la tècnica d'early stopping. 
def check_early_stopping(state, val_loss):
    # Comprova si és la primera vegada o si la pèrdua de validació ha millorat per sobre del mínim canvi requerit
    if state["best_loss"] is None or val_loss < state["best_loss"] - state["min_delta"]:
        # Actualitza la millor pèrdua registrada i reinicia el comptador
        state["best_loss"] = val_loss
        state["counter"] = 0
    else:
        # Incrementa el comptador si no hi ha hagut millora en la pèrdua de validació
        state["counter"] += 1
        
        # Si el comptador arriba a la paciència establerta, activa l'early stopping
        if state["counter"] >= state["patience"]:
            state["early_stop"] = True
    
    # Retorna si s'ha d'aturar l'entrenament o no
    return state["early_stop"]

# Entrena el model per una sola epoch utilitzant les dades del train_loader
def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    # Configura el model en mode d'entrenament (això activa operacions com el Dropout)
    model.train()
    
    # Variables per seguir la pèrdua total, el nombre de prediccions correctes i el nombre total de mostres
    total_loss = 0
    correct = 0
    total = 0
    
    # Barra de progrés per visualitzar l'avanç de l'entrenament a cada epoch
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]")
    for inputs, labels in pbar:
        # Moure les dades d'entrada i les etiquetes al dispositiu (CPU o GPU)
        inputs = inputs.to(device)
        # Convertir a float per BCE (Binary Cross Entropy)
        labels = labels.float().to(device)  
        
        # Reinicialitzar els gradients dels paràmetres optimitzables
        optimizer.zero_grad()
        
        # Passar les dades pel model per obtenir les prediccions
        outputs = model(inputs).squeeze()
        
        # Calcular la pèrdua entre les prediccions i les etiquetes reals
        loss = criterion(outputs, labels)
        
        # Fer la retropropagació per calcular els gradients
        loss.backward()
        
        # Actualitzar els paràmetres del model utilitzant l'optimitzador
        optimizer.step()
        
        # Actualitzar la pèrdua total per a l'epoch
        total_loss += loss.item()
        
        # Convertir les sortides a prediccions binàries (0 o 1) utilitzant un llindar de 0.5
        predicted = (outputs > 0.5).float()
        
        # Comprovar el nombre de prediccions correctes
        correct += (predicted == labels).sum().item()
        
        # Actualitzar el total de mostres processades
        total += labels.size(0)
        
        # Actualitzar la barra de progrés amb la pèrdua mitjana i l'exactitud actuals
        pbar.set_postfix({
            'loss': f'{total_loss/len(train_loader):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    # Retornar la pèrdua mitjana i l'exactitud de l'epoch
    return total_loss/len(train_loader), 100.*correct/total

# S'encarrega d'avaluar el rendiment del model utilitzant un conjunt de dades de validació
# La validació és un pas crític en el procés d'entrenament perquè
# - Avaluar el Rendiment General del Model
# - Evitar el Sobreajustament (Overfitting)
def validate(model, val_loader, criterion, device, epoch):
    # Configura el model en mode d'avaluació (això desactiva operacions com Dropout)
    model.eval()
    
    # Variables per seguir la pèrdua total, el nombre de prediccions correctes i el nombre total de mostres
    total_loss = 0
    correct = 0
    total = 0
    
    # Desactiva el càlcul dels gradients, ja que no es necessita durant l'avaluació
    with torch.no_grad():
        # Barra de progrés per visualitzar l'avanç de l'avaluació
        pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]")
        for inputs, labels in pbar:
            # Moure les dades d'entrada i les etiquetes al dispositiu (CPU o GPU)
            inputs = inputs.to(device)
            labels = labels.float().to(device)
            
            # Passar les dades pel model per obtenir les prediccions
            outputs = model(inputs).squeeze()
            
            # Calcular la pèrdua entre les prediccions i les etiquetes reals
            loss = criterion(outputs, labels)
            
            # Actualitzar la pèrdua total per a l'epoch
            total_loss += loss.item()
            
            # Convertir les sortides a prediccions binàries (0 o 1) utilitzant un llindar de 0.5
            predicted = (outputs > 0.5).float()
            
            # Comprovar el nombre de prediccions correctes
            correct += (predicted == labels).sum().item()
            
            # Actualitzar el total de mostres processades
            total += labels.size(0)
            
            # Actualitzar la barra de progrés amb la pèrdua mitjana i l'exactitud actuals
            pbar.set_postfix({
                'loss': f'{total_loss/len(val_loader):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    # Retornar la pèrdua mitjana i l'exactitud de l'epoch
    return total_loss/len(val_loader), 100.*correct/total

def main():
    # Configurar el dispositiu (GPU si està disponible, sinó CPU)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" or device.type == "mps":
        print(f"Using device: {device} (GPU accelerated)")
    else:
        print(f"Using device: {device} (CPU based)")
    
    # Crear DataLoaders per a entrenament i validació, i obtenir les classes del dataset
    train_loader, val_loader, classes = create_data_loaders()
    print(f"Classes: {classes}")
    
    # Afegir les classes a la configuració i guardar-la en un fitxer per referències futures
    config['classes'] = classes
    with open(config['config_path'], "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    
    # Crear el model i moure'l al dispositiu (GPU/CPU)
    model = create_model()
    model = model.to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad)} weigths and biases")
    
    # Definir la funció de pèrdua per classificació binària
    criterion = nn.BCELoss()  # Binary Cross Entropy
    
    # Configurar l'optimitzador AdamW amb una taxa d'aprenentatge definida
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=0.01)
    
    # Configurar el Scheduler per reduir la taxa d'aprenentatge basant-se en la pèrdua de validació
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode=config['reduce_lr_on_plateau']['mode'], 
        factor=config['reduce_lr_on_plateau']['factor'], 
        patience=config['reduce_lr_on_plateau']['patience'])
    
    # Inicialitzar l'estat per a l'early stopping
    early_stopping_state = initialize_early_stopping()
    
    # Variable per seguir la millor exactitud de validació aconseguida
    best_val_acc = 0.0
    
    # Bucle principal d'entrenament/validació per a cada epoch
    for epoch in range(EPOCHS):
        # Entrenar el model per una epoch
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validar el model amb el conjunt de validació
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch)
        
        # Imprimir les mètriques d'entrenament i validació, així com la taxa d'aprenentatge actual
        print(f"""Epoch {epoch}/{EPOCHS} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - LR: {optimizer.param_groups[0]['lr']:.6f} """)
        
        # Actualitzar el scheduler segons la pèrdua de validació per ajustar la taxa d'aprenentatge
        scheduler.step(val_loss)
        
        # Guardar el model si s'obté una millor exactitud de validació
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"Saving a better model with accuracy {val_acc:.2f}%")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, config['model_path'])
        
        # Comprovar si cal activar l'early stopping i aturar l'entrenament
        if check_early_stopping(early_stopping_state, val_loss):
            print("Early stopping activated")
            break

if __name__ == "__main__":
    main()