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

# TODO: Resta del codi