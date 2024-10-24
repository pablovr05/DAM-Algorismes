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
    [f"{DATA_FOLDER}/img10042414.jpg", "smile"],
    [f"{DATA_FOLDER}/img11599296.jpg", "non_smile"],
    # Acabar d'afegir i classificar totes les cares de ./data/test
]

# Carregar configuració
with open('iscat_config.json', 'r') as f:
    config = json.load(f)

# TODO: Resta del codi