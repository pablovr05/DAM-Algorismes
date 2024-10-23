#!/usr/bin/env python3

from PIL import Image
import os

image_dir = './data'
for subdir, _, files in os.walk(image_dir):
    print(f"Folder: {subdir}")
    for file in files:
        if file.endswith('.png') or file.endswith('.jpg'):
            image_path = os.path.join(subdir, file)
            with Image.open(image_path) as img:
                img = img.convert("RGB")  # Convertir a RGB si cal
                img.save(image_path, icc_profile=None)