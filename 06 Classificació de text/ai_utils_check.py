#!/usr/bin/env python3

import sys
import subprocess

required_libraries = [
    "torch",
    "torchvision",
    "torchaudio",
    "Pillow",
    "tqdm",
    "pandas",
    "numpy",
    "scikit-learn",
    "transformers"
]

def check_and_suggest_installation():
    missing_libraries = []
    for lib in required_libraries:
        try:
            subprocess.check_output([sys.executable, "-m", "pip", "show", lib])
        except subprocess.CalledProcessError:
            missing_libraries.append(lib)
    
    if missing_libraries:
        print("Les següents llibreries no estan instal·lades:", ", ".join(missing_libraries))
        install_command = f"python3 -m pip install {' '.join(missing_libraries)} --break-system-package"
        print("Instrucció per instal·lar-les:")
        print(install_command)
        sys.exit(1)

check_and_suggest_installation()

import torch

def check_device_and_suggest():
    try:
        if torch.cuda.is_available():
            if torch.version.hip:
                print("Suggeriment: El sistema sembla suportar AMD ROCm. Assegura't de tenir ROCm instal·lat.")
                print("Per instal·lar ROCm, consulta: https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html")
                print("Instrucció pip per instal·lar PyTorch amb suport ROCm:")
                print("pip install torch torchvision torchaudio Pillow tqdm --extra-index-url https://download.pytorch.org/whl/rocm5.4.2")
            else:
                print("Suggeriment: El sistema sembla suportar NVIDIA CUDA. Assegura't de tenir CUDA i cuDNN instal·lats.")
                print("Descarrega CUDA Toolkit: https://developer.nvidia.com/cuda-downloads")
                print("Descarrega cuDNN: https://developer.nvidia.com/cudnn")
                print("Instrucció pip per instal·lar PyTorch amb suport CUDA:")
                print("pip install torch torchvision torchaudio Pillow tqdm --extra-index-url https://download.pytorch.org/whl/cu118")
        elif torch.backends.mps.is_available():
            print("Suggeriment: El sistema sembla suportar MPS (Apple Silicon).")
            print("Instrucció pip per instal·lar PyTorch amb suport MPS (Apple Silicon):")
            print("python3 -m pip install torch torchvision torchaudio Pillow tqdm --break-system-package")
        elif sys.platform.startswith('linux'):
            print("Suggeriment: Per a acceleració amb NVIDIA, instal·la CUDA i cuDNN.")
            print("Descarrega CUDA Toolkit: https://developer.nvidia.com/cuda-downloads")
            print("Descarrega cuDNN: https://developer.nvidia.com/cudnn")
            print("Instrucció pip per instal·lar PyTorch amb suport CUDA:")
            print("pip install torch torchvision torchaudio Pillow tqdm --extra-index-url https://download.pytorch.org/whl/cu118")
            print("Suggeriment: Per a acceleració amb AMD, assegura't de tenir ROCm compatible instal·lat.")
            print("Consulta la guia d'instal·lació de ROCm: https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html")
            print("Instrucció pip per instal·lar PyTorch amb suport ROCm:")
            print("pip install torch torchvision torchaudio Pillow tqdm --extra-index-url https://download.pytorch.org/whl/rocm5.4.2")
        elif sys.platform == "darwin":
            print("Suggeriment: Per a acceleració amb Apple Silicon, assegura't de tenir PyTorch instal·lat amb suport MPS.")
            print("Instrucció pip per instal·lar PyTorch amb suport MPS (Apple Silicon):")
            print("python3 -m pip install torch torchvision torchaudio Pillow tqdm --break-system-package")
        elif sys.platform.startswith('win'):
            print("Suggeriment: Per a acceleració amb NVIDIA en Windows, instal·la CUDA Toolkit i cuDNN.")
            print("Descarrega CUDA Toolkit: https://developer.nvidia.com/cuda-downloads")
            print("Descarrega cuDNN: https://developer.nvidia.com/cudnn")
            print("Instrucció pip per instal·lar PyTorch amb suport CUDA:")
            print("pip install torch torchvision torchaudio Pillow tqdm --extra-index-url https://download.pytorch.org/whl/cu118")
    except Exception as e:
        print(f"Error al verificar l'entorn: {e}")

device = torch.device("cpu")

if torch.cuda.is_available():
    if torch.version.hip:
        device = torch.device("cuda") 
        print("AMD ROCm GPU is available. Using:", device)
    else:
        device = torch.device("cuda") 
        print("NVIDIA CUDA GPU is available. Using:", device)
elif torch.backends.mps.is_available():
    device = torch.device("mps") 
    print("MPS (Metal) GPU is available. Using:", device)
elif hasattr(torch.backends, 'opencl') and torch.backends.opencl.is_available():
    device = torch.device("opencl")
    print("OpenCL GPU is available. Using:", device)
else:
    print("No GPU available. Using:", device)
    check_device_and_suggest()

x = torch.tensor([1.0, 2.0, 3.0], device=device)
print(f"Tensor on: {x.device}")
