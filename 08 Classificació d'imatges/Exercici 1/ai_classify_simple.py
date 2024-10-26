#!/usr/bin/env python3

# python3 -m pip install torch torchvision torchaudio Pillow tqdm --break-system-package

import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

model_path = 'species_model.pth'
types_path = 'species_types.json'

# TODO: Test images
