#!/usr/bin/env python3

import os
import json
import shutil
import random
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer

DATA_FOLDER = './data/'

# Carregar configuració
with open('engorother_config.json', 'r') as f:
    config = json.load(f)

# TODO: Rest of the code