#!/usr/bin/env python3

import os
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Configuration
config = {
    "config_path": "engorother_config.json",
    "model_path": "engorother_model.pth",
    "vocab_path": "engorother_vocab.json",
    "csv_path": "./data/engorother.csv",
    "training": {
        "batch_size": 32,
        "epochs": 25,
        "learning_rate": 0.001,
        "validation_split": 0.2
    },
    "model_params": {
        "hidden_size": 256,
        "dropout_rate": 0.5,
        "num_output": 1
    },
    "early_stopping": {
        "patience": 5,
        "min_delta": 0
    },
    "reduce_lr_on_plateau": {
        "mode": "min",
        "factor": 0.1,
        "patience": 3
    },
    "classes": []
}

# TODO: Rest of the code