#!/usr/bin/env python3

# python3 -m pip install torch torchvision torchaudio Pillow tqdm --break-system-package

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm

model_path = 'species_model.pth'
types_path = 'species_types.json'

# TODO: Train model ...