#!/usr/bin/env python3

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm

# Constants
MODEL_PATH = 'face_model.pth'
TYPES_PATH = 'face_classes.json'

