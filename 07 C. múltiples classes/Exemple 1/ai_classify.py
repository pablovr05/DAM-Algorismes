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
with open('news_config.json', 'r') as f:
    config = json.load(f)

# Selecciona 50 missatges aleatoris del CSV
def load_random_test_texts(csv_path, num_samples=50):
    df = pd.read_csv(csv_path)
    
    # Utilitzar les columnes correctes del dataset
    column_class = config['csv_cloumn_names']['class']
    column_text = config['csv_cloumn_names']['text']
    texts = df[[column_text, column_class]].rename(columns={'body': column_text, 'category': column_class})
    
    # Seleccionar 50 mostres aleatòries
    test_texts = texts.sample(n=num_samples).values.tolist()
    return test_texts

# Crea una nova instància del model TextClassifier
def create_text_classifier_model(input_size, hidden_size, dropout_rate, num_classes):
    """Crear el mateix model que es va utilitzar per entrenar"""
    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(hidden_size, hidden_size // 2),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(hidden_size // 2, num_classes),
        nn.Softmax(dim=1)  # Per múltiples classes
    )
    return model


def load_model(model_path, input_size, hidden_size, dropout_rate, num_classes, device):
    """Carregar el model amb els pesos entrenats"""
    # Crear una instància del model classificació de text
    model = create_text_classifier_model(input_size, hidden_size, dropout_rate, num_classes)
    
    # Carregar el diccionari complet
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Extreure només l'estat del model
    state_dict = checkpoint['model_state_dict']
    
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
    correct = 0
    total = 0
    predictions = []
    
    with torch.no_grad():
        for text, true_label in tqdm(test_texts, disable=True):
            text_vector = torch.FloatTensor(
                vectorizer.transform([text]).toarray()
            ).squeeze().unsqueeze(0).to(device)
            
            outputs = model(text_vector)
            
            # Les probabilitats de sortida són directament les sortides del model després del Softmax
            probabilities = outputs.squeeze().cpu().numpy()
            
            # Determinar l'etiqueta predita i la probabilitat màxima
            predicted_class = torch.argmax(outputs, 1).item()
            predicted_label = class_names[predicted_class]
            confidence = probabilities[predicted_class]
            
            # Comprovar si la predicció és correcta
            is_correct = predicted_label == true_label
            if is_correct:
                correct += 1
            total += 1
            
            predictions.append({
                'text': text,
                'predicted': predicted_label,
                'true_label': true_label,
                'confidence': confidence,
                'correct': is_correct
            })
            
            print(f"Text: {text[:40].ljust(40)}..., Prediction: {f'{confidence:.2%}'.rjust(8)} = {"'"+predicted_label+"'":6} ({"'"+true_label+"'":6} > {'correct' if is_correct else 'wrong'})")
            
            # Mostrem les probabilitats per cada classe
            probs_str = " ,".join([f"{class_names[i][:5]}:{prob*100:5.2f}%" for i, prob in enumerate(probabilities)])
            print(f"{probs_str}")

    return correct, total

def main():

    # Configurar el dispositiu (GPU si està disponible, sinó CPU)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" or device.type == "mps":
        print(f"Using device: {device} (GPU accelerated)")
    else:
        print(f"Using device: {device} (CPU based)")
    
    class_names = config['classes']
    print(f"Classes: {class_names}")
    
    test_texts = load_random_test_texts(config['csv_path'])
    
    with open(config['vocab_path'], 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    vectorizer = CountVectorizer(vocabulary=vocab_data['vocabulary'])
    input_size = len(vectorizer.vocabulary)
    num_classes = len(class_names)

    model = load_model(config['model_path'], input_size, config['model_params']['hidden_size'], config['model_params']['dropout_rate'], num_classes, device)
    
    correct, total = evaluate_model(model, test_texts, vectorizer, class_names, device)
    
    accuracy = correct / total
    print("\nGlobal results:")
    print(f"Total texts: {total}")
    print(f"Hits: {correct}")
    print(f"Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()
