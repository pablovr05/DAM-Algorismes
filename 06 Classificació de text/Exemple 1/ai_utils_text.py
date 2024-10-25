import numpy as np
import torch
import torch.nn as nn
import json
import pandas as pd
from torch.utils.data import DataLoader, random_split, TensorDataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

def create_model_from_config(config, num_categories):
    layers = []
    input_size = config['model_definition']['input_size']
    
    for layer_config in config['model_definition']['layers']:
        if layer_config['type'] == "Linear":
            output_size = layer_config['size']
            if output_size == "num_categories":
                output_size = num_categories
            
            layers.append(nn.Linear(input_size, output_size))
            
            if 'activation' in layer_config and layer_config['activation'] == "ReLU":
                layers.append(nn.ReLU())
            
            if 'dropout' in layer_config:
                layers.append(nn.Dropout(layer_config['dropout']))
            
            input_size = output_size

    return nn.Sequential(*layers)

def adjust_input_shape(vectorized_data, target_size=5000):
    if vectorized_data.shape[1] < target_size:
        padding = np.zeros((vectorized_data.shape[0], target_size - vectorized_data.shape[1]))
        vectorized_data = np.hstack((vectorized_data, padding))
    return vectorized_data

def create_data_loaders(config):
    print("Loading dataset...")
    
    # Read CSV file
    df = pd.read_csv(config['paths']['data'])
    
    # Identify unique categories and create label encoder
    column_categories = config['columns']['categories']
    column_text = config['columns']['text']
    unique_categories = sorted(df[column_categories].unique())
    label_encoder = {cat: idx for idx, cat in enumerate(unique_categories)}
    
    # Convert labels using the encoder
    labels = df[column_categories].map(label_encoder).values
    texts = df[column_text].values

    # Create and fit the vectorizer
    vectorizer = CountVectorizer(max_features=config['pre_processing']['max_features'])
    vectorized_texts = vectorizer.fit_transform(texts).toarray()

    # Adjust input shape if needed
    vectorized_texts = adjust_input_shape(vectorized_texts, config['model_definition']['input_size'])
    
    # Split data into train and validation sets (manca aquesta part)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        vectorized_texts, labels, 
        test_size=config['training']['validation_split'], 
        random_state=42
    )
        
    # Convert to PyTorch tensors
    train_texts = torch.tensor(train_texts, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    val_texts = torch.tensor(val_texts, dtype=torch.float32)
    val_labels = torch.tensor(val_labels, dtype=torch.long)
    
    # Create Tensor datasets
    train_dataset = torch.utils.data.TensorDataset(train_texts, train_labels)
    val_dataset = torch.utils.data.TensorDataset(val_texts, val_labels)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    print(f"Dataset loaded: {len(train_texts)} training articles, {len(val_texts)} validation articles")
    print(f"Categories: {', '.join(unique_categories)}")
    
    return unique_categories, train_loader, val_loader, vectorizer, label_encoder

def initialize_early_stopping(config):
    return {
        "patience": config['optimization']['early_stopping']['patience'],
        "min_delta": config['optimization']['early_stopping']['min_delta'],
        "counter": 0,
        "best_loss": None,
        "early_stop": False
    }

def load_config(path_config, mode="train"):
    # Load the configuration from file
    with open(path_config, 'r') as f:
        config = json.load(f)
    
    # Determine the device (MPS, CUDA, or CPU)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" or device.type == "mps":
        print(f"Using device: {device} (GPU accelerated)")
    else:
        print(f"Using device: {device} (CPU based)")
    
    if mode == "train":
        # Create data loaders 
        unique_categories, train_loader, val_loader, vectorizer, label_encoder = create_data_loaders(config)       
        
        # Create model
        model = create_model_from_config(config, num_categories=len(unique_categories))
        model.to(device)
        
        # Initialize early stopping
        early_stopping_state = initialize_early_stopping(config)
        
        return config, unique_categories, model, device, train_loader, val_loader, vectorizer, label_encoder, early_stopping_state
    
    elif mode == "classify":
        # Load model state dict from the provided path
        model_path = config['paths']['trained_network']
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        
        # Load config from the checkpoint to ensure consistency
        loaded_config = checkpoint.get('config', config)

        # Load metadata
        with open(loaded_config['paths']['metadata'], 'r') as f:
            metadata = json.load(f)
            unique_categories = metadata['unique_categories']
            label_encoder = metadata['label_encoder']
            vocab = metadata['vocabulary']
        
        # Create vectorizer from saved vocabulary
        vectorizer = CountVectorizer(vocabulary=vocab)
        
        # Create model from config and load state dict
        model = create_model_from_config(loaded_config, num_categories=len(unique_categories))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()  # Set model to evaluation mode
        
        return config, model, device, label_encoder, vectorizer
    
def save_metadata(unique_categories, vectorizer, label_encoder, metadata_path):
    # Save vocabulary and label encoder
    with open(metadata_path, 'w') as f:
        json.dump({
            'unique_categories': unique_categories,
            'label_encoder': label_encoder,
            'vocabulary': {k: int(v) for k, v in vectorizer.vocabulary_.items()}
        }, f)

def save_model(model, label_encoder, model_path):
    print(f"Saving model to {model_path}")
    torch.save({
        'model_state_dict': model.state_dict(),
        'label_encoder': label_encoder
    }, model_path)