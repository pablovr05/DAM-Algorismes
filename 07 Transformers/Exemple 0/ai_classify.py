#!/usr/bin/env python3

import os
print("Loading AI libraries ..."); import torch
import pandas as pd
from ai_utils_text import load_config, adjust_input_shape

CONFIG_FILE = "model_config.json"

def clearScreen():
    if os.name == 'nt':     # Si estàs a Windows
        os.system('cls')
    else:                   # Si estàs a Linux o macOS
        os.system('clear')

clearScreen()

def classify_texts(texts, model, vectorizer, device):
    # Convert texts to vectorized input using the same vocabulary
    vectorized_texts = vectorizer.transform(texts).toarray()
    
    # Adjust input shape if needed (mantenir compatibilitat amb entrenament)
    vectorized_texts = adjust_input_shape(vectorized_texts, 5000)
    
    inputs = torch.tensor(vectorized_texts, dtype=torch.float32).to(device)
    
    # Run the model for classification
    model.eval()  # Ensure model is in evaluation mode
    with torch.no_grad():
        outputs = model(inputs).squeeze()  # Ajusta la sortida perquè tingui la forma adequada
        probabilities = torch.sigmoid(outputs)  # Converteix els logits a probabilitats
        predicted = (probabilities >= 0.5).int()  # Classifica com a 1 si >= 0.5, sinó 0
    
    return predicted.cpu().numpy(), probabilities.cpu().numpy()

def main():
    # Load the configuration and model for classification
    config, model, device, label_encoder, vectorizer = load_config(CONFIG_FILE, mode="classify")
    
    # Load dataset and select 50 random texts
    df = pd.read_csv(config['paths']['data'])
    column_categories = config['columns']['categories']
    column_text = config['columns']['text']
    sample_df = df.sample(50)

    true_labels = sample_df[column_categories].values    
    texts = sample_df[column_text].values
    
    # Decode labels using the saved label_encoder
    label_decoder = {v: k for k, v in label_encoder.items()}
    class_names = list(label_decoder.values())
    
    # Classify the texts
    predicted_labels, probabilities = classify_texts(texts, model, vectorizer, device)
    
    # Initialize counters
    correct = 0
    total = len(texts)
    
    # Display results for each text
    for i, text in enumerate(texts):
        predicted_label = label_decoder[predicted_labels[i]]
        true_label = true_labels[i]
        is_correct = (predicted_label == true_label)
        if is_correct:
            correct += 1
        
        confidence = probabilities[i]  # Probabilitat de la classe positiva
        print(f"\nText: {text[:50].ljust(50)}..., Prediction: {f'{confidence:.2%}'.rjust(6)} = {'"'+predicted_label+'"':6} ({'"'+true_label+'"':6} > {'correct' if is_correct else 'wrong'})")
        
        # Mostra la probabilitat de la classificació
        print(f"Probabilitat de classe positiva: {confidence:.2f}")
    
    # Display global statistics
    accuracy = correct / total
    print("\nGlobal results:")
    print(f"  Total texts: {total}")
    print(f"  Hits: {correct}")
    print(f"  Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()

