#!/usr/bin/env python3

import os
print("Loading AI libraries ..."); import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ai_utils_text import load_config

CONFIG_FILE = "model_config.json"

def clearScreen():
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')

clearScreen()

def classify_texts(texts, model, vectorizer, device):
    vectorized_texts = vectorizer.transform(texts).toarray()
    inputs = torch.tensor(vectorized_texts, dtype=torch.float32).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        probabilities = torch.sigmoid(outputs)
    predicted = (probabilities > 0.5).int()
    
    return predicted.cpu().numpy(), probabilities.cpu().numpy()

def get_true_labels(df_data, df_relation, df_labels, label_encoder):
    tag_map = df_labels.set_index('tag_id')['tag_name'].to_dict()
    df_relation['tag_name'] = df_relation['tag_id'].map(tag_map)
    book_tags = df_relation.groupby('goodreads_book_id')['tag_name'].apply(list).to_dict()
    df_data['tags'] = df_data['best_book_id'].map(book_tags)
    
    true_binary_labels = []
    for tags in df_data['tags'].fillna(''):
        if isinstance(tags, str):
            tags = []
        binary_label = [0] * len(label_encoder)
        for tag in tags:
            if tag in label_encoder:
                binary_label[label_encoder[tag]] = 1
        true_binary_labels.append(binary_label)
    
    return true_binary_labels

def calculate_jaccard_similarity(predicted_tags, true_tags):
    predicted_set = set(predicted_tags)
    true_set = set(true_tags)
    intersection = len(predicted_set.intersection(true_set))
    union = len(predicted_set.union(true_set))
    if union == 0:
        return 0
    return intersection / union

def main():
    config, model, device, label_encoder, vectorizer = load_config(CONFIG_FILE, mode="classify", multi_label=True)
    
    df_data = pd.read_csv(config['paths']['data'])
    df_labels = pd.read_csv(config['paths']['data_labels'])
    df_relation = pd.read_csv(config['paths']['data_relation'])
    
    column_text = "title"
    sample_df = df_data.sample(50)
    texts = sample_df[column_text].values
    
    true_binary_labels = get_true_labels(sample_df, df_relation, df_labels, label_encoder)
    true_binary_labels = torch.tensor(true_binary_labels, dtype=torch.int).numpy()
    
    label_decoder = {v: k for k, v in label_encoder.items()}
    class_names = list(label_decoder.values())
    
    predicted_labels, probabilities = classify_texts(texts, model, vectorizer, device)
    
    accuracy = accuracy_score(true_binary_labels, predicted_labels)
    precision = precision_score(true_binary_labels, predicted_labels, average='micro')
    recall = recall_score(true_binary_labels, predicted_labels, average='micro')
    f1 = f1_score(true_binary_labels, predicted_labels, average='micro')

    exact_matches = 0
    jaccard_sum = 0

    print(f"\nGlobal Statistics:")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  Precision: {precision:.2%}")
    print(f"  Recall: {recall:.2%}")
    print(f"  F1 Score: {f1:.2%}")

    for i, text in enumerate(texts):
        predicted_tags = [class_names[j] for j in range(len(class_names)) if predicted_labels[i][j] == 1]
        true_tags = [class_names[j] for j in range(len(class_names)) if true_binary_labels[i][j] == 1]
        confidence_tags = [(class_names[j], probabilities[i][j]) for j in range(len(class_names)) if predicted_labels[i][j] == 1]

        if set(predicted_tags) == set(true_tags):
            exact_matches += 1

        jaccard_similarity = calculate_jaccard_similarity(predicted_tags, true_tags)
        jaccard_sum += jaccard_similarity

        print(f"\nText: {text[:50].ljust(50)}...")
        print(f"Predicted Tags: {', '.join(predicted_tags)}")
        print(f"True Tags: {', '.join(true_tags)}")
        confidence_str = ", ".join([f"{tag}: {confidence:.2%}" for tag, confidence in confidence_tags])
        print(f"Confidences: {confidence_str}")
        print(f"Jaccard Similarity: {jaccard_similarity:.2%}")

    exact_match_percentage = (exact_matches / len(texts)) * 100
    average_jaccard_similarity = (jaccard_sum / len(texts)) * 100

    print("\nGlobal Results:")
    print(f"  Total texts: {len(texts)}")
    print(f"  Exact Matches: {exact_matches}")
    print(f"  Exact Match Percentage: {exact_match_percentage:.2f}%")
    print(f"  Average Jaccard Similarity: {average_jaccard_similarity:.2f}%")

if __name__ == "__main__":
    main()
