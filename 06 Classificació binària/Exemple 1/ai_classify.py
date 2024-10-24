#!/usr/bin/env python3

import os
import json
import shutil
import random
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer

DATA_FOLDER = './data/'

# Carregar configuració
with open('hamorspam_config.json', 'r') as f:
    config = json.load(f)

# Selecciona 50 missatges aleatoris del CSV
def load_random_test_texts(csv_path, num_samples=50):
    df = pd.read_csv(csv_path)
    
    # Convertir etiquetes a "ham" o "spam" segons l'arxiu de configuració
    texts = df[['v2', 'v1']].rename(columns={'v2': 'text', 'v1': 'label'})
    
    # Seleccionar 50 mostres aleatòries
    test_texts = texts.sample(n=num_samples).values.tolist()
    return test_texts

# Crea una nova instància del model TextClassifier
def create_text_classifier_model(input_size, hidden_size, dropout_rate):
    """Crear el mateix model que es va utilitzar per entrenar"""
    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(hidden_size, hidden_size // 2),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(hidden_size // 2, 1),
        nn.Sigmoid()
    )
    return model

def load_model(model_path, input_size, hidden_size, dropout_rate, device):
    """Carregar el model amb els pesos entrenats"""
    # Crear una instància del model classificació de text
    model = create_text_classifier_model(input_size, hidden_size, dropout_rate)
    
    # Carregar només els pesos del model
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    
    # Ajustar les claus del `state_dict` si comencen amb "model."
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    
    # Carregar l'estat del model des del `state_dict` carregat
    model.load_state_dict(state_dict)
    
    # Moure el model al dispositiu especificat
    model = model.to(device)
    
    # Configurar el model en mode d'avaluació (desactivant Dropout, BatchNorm, etc.)
    model.eval()
    
    return model


# Avaluar el rendiment del model utilitzant un conjunt de missatges de test
def evaluate_model(model, test_texts, vectorizer, class_names, device):
    # Variables per seguir el nombre de prediccions correctes i el total de missatges
    correct = 0
    total = 0
    predictions = []
    
    # Desactiva el càlcul dels gradients perquè només es vol fer inferència
    with torch.no_grad():
        for text, true_label in tqdm(test_texts, disable=True):
            # Vectoritzar el text
            text_vector = torch.FloatTensor(
                vectorizer.transform([text]).toarray()
            ).squeeze().unsqueeze(0).to(device)
            
            # Fer predicció amb el model
            outputs = model(text_vector)
            
            # Convertir la sortida del model a una probabilitat
            predicted_prob = outputs.item()
            
            # Determinar l'etiqueta predita basada en un llindar de 0.5
            predicted_label = class_names[1] if predicted_prob > 0.5 else class_names[0]
            
            # Calcular la confiança en la predicció
            confidence = predicted_prob if predicted_prob > 0.5 else (1 - predicted_prob)
            
            # Comprovar si la predicció és correcta
            is_correct = predicted_label == true_label
            if is_correct:
                correct += 1
            total += 1
            
            # Guardar els resultats de la predicció
            predictions.append({
                'text': text,
                'predicted': predicted_label,
                'true_label': true_label,
                'confidence': confidence,
                'correct': is_correct
            })
            
            # Mostrar resultat en consola amb format clar
            print(f"Text: {text[:30].ljust(30)}..., Prediction: {f'{confidence:.2%}'.rjust(8)} = {"'"+predicted_label+"'":6} ({"'"+true_label+"'":6} > {'correct' if is_correct else 'wrong'})")    
    
    return correct, total

def main():
    # Configurar dispositiu
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" or device.type == "mps":
        print(f"Using device: {device} (GPU accelerated)")
    else:
        print(f"Using device: {device} (CPU based)")
    
    # Carregar les classes des de la configuració
    class_names = config['classes']
    print(f"Classes: {class_names}")
    
    # Carregar 50 missatges aleatoris del CSV per a test
    test_texts = load_random_test_texts(config['csv_path'])
    
    # Carregar vectorizer del fitxer guardat
    with open(config['vocab_path'], 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    vectorizer = CountVectorizer(vocabulary=vocab)
    
    # Preparar el model
    input_size = len(vectorizer.vocabulary)
    model = load_model(config['model_path'], input_size, config['model_params']['hidden_size'], config['model_params']['dropout_rate'], device)
    
    # Avaluar el model utilitzant els missatges de test
    correct, total = evaluate_model(
        model, test_texts, vectorizer, class_names, device
    )
    
    # Calcular i mostrar les mètriques de rendiment
    accuracy = correct / total
    print("\nGlobal results:")
    print(f"Total texts: {total}")
    print(f"Hits: {correct}")
    print(f"Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()
