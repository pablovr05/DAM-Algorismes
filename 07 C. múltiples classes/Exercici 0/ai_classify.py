#!/usr/bin/env python3

import os
import json
import shutil
import zipfile
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
from pathlib import Path
from tqdm import tqdm

DATA_FOLDER = './data/testing'

# Imatges de test amb les seves etiquetes
test_images = [
    [f"{DATA_FOLDER}/im832E2E543DD946A797309D12A95D8697.jpeg", "squirrel"],
    [f"{DATA_FOLDER}/im351744278DA544B6A1365A3EA33880E3.jpg", "sheep"]
    # TODO: Rest of the testing images
]

# Carregar configuració
with open('species_config.json', 'r') as f:
    config = json.load(f)

# TODO: Rest of the code