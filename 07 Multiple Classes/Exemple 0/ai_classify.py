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
    [f"{DATA_FOLDER}/0abad4440c4415b9707f11151762526c.jpg", "racing"],
    [f"{DATA_FOLDER}/0ea9451a8de9cf7581835ccb20940174.jpg", "family"],
    [f"{DATA_FOLDER}/1a3f888928097b855e49c1926b94a5dc.jpg", "truck"],
    [f"{DATA_FOLDER}/1b97666eaae94e70cba5e238017cab7a.jpg", "truck"],
    [f"{DATA_FOLDER}/1ded9b835b61b963f13d6b4374ef2492.jpg", "jeep"],
    [f"{DATA_FOLDER}/2b6da117b6b37e8fa1d48cddec05b5b2.jpg", "racing"],
    [f"{DATA_FOLDER}/2b96ccd24db32c2c47581afde8d42722.jpg", "bus"],
    [f"{DATA_FOLDER}/2f313143d05ede99c2458ba697161c1c.jpg", "truck"],
    [f"{DATA_FOLDER}/3acbdbd78d3270f626fa9eac8c182bb3.jpg", "family"],
    [f"{DATA_FOLDER}/3db10fa5700a75911c64410f35d7b94d.jpg", "racing"],
    [f"{DATA_FOLDER}/3dbb4fcf817656a9a865f3ea33f07a6a.jpg", "racing"],
    [f"{DATA_FOLDER}/3f7a8b1ae0bc3d5d82e1240f810ce47d.jpg", "truck"],
    [f"{DATA_FOLDER}/4aa43e160ee667c2deb922cd371204bc.jpg", "jeep"],
    [f"{DATA_FOLDER}/4b71b5a2fa94a71b0581c6650a0f8d0a.jpg", "truck"],
    [f"{DATA_FOLDER}/4baa8146de477ebafae4b3f46d5614c2.jpg", "truck"],
    [f"{DATA_FOLDER}/05a006ebd8667a39988a253b10956bb7.jpg", "racing"],
    [f"{DATA_FOLDER}/5d7514af404266c3f92efcee8662f640.jpg", "bus"],
    [f"{DATA_FOLDER}/5edeade7ca9e85efa46e1be268be129c.jpg", "truck"],
    [f"{DATA_FOLDER}/6a4e8af47c5b5b1412ce9a28a247d50e.jpg", "family"],
    [f"{DATA_FOLDER}/6f13f51f98bd1e2332bfa6f1c1c5ff14.jpg", "jeep"],
    [f"{DATA_FOLDER}/7a9027a09a9149fd901f97630a052b29.jpg", "taxi"],
    [f"{DATA_FOLDER}/7fe25c5c90c4f447fe3fa10adf4232cd.jpg", "bus"],
    [f"{DATA_FOLDER}/09bbbd3f758e129a2692844e3b62d5a8.jpg", "family"],
    [f"{DATA_FOLDER}/9cecedb8bfe67dbbc76d7723c3b4de43.jpg", "truck"],
    [f"{DATA_FOLDER}/9e37ae8bf11600a30afcd592d73a4be1.jpg", "bus"],
    [f"{DATA_FOLDER}/34d68f0a8eda8aadc568c1fd14b222b9.jpg", "family"],
    [f"{DATA_FOLDER}/37d329f08d105885c3e5d658921356c6.jpg", "family"],
    [f"{DATA_FOLDER}/48dc3c3c0038b87e330d7692cae4b0ff.jpg", "family"],
    [f"{DATA_FOLDER}/51a610fa467c9b6d0a0c9ebc87ebd6cd.jpg", "truck"],
    [f"{DATA_FOLDER}/61b29def40ee024890ebc2af8851a120.jpg", "truck"],
    [f"{DATA_FOLDER}/63f9f7dc8298317284bf7dbf2074b3ba.jpg", "bus"],
    [f"{DATA_FOLDER}/75f2660d690d5bc035c1f9d9424e253a.jpg", "truck"],
    [f"{DATA_FOLDER}/81eba54e555030c678afe71bb81c3480.jpg", "jeep"],
    [f"{DATA_FOLDER}/84c8aadbabbeb753042593faffa4e376.jpg", "truck"],
    [f"{DATA_FOLDER}/95ac3b0debba83ee706b4edef03ee2c2.jpg", "truck"],
    [f"{DATA_FOLDER}/98b83106dcaeeb585913fb59c8525757.jpg", "family"],
    [f"{DATA_FOLDER}/210d3833b9c3b0613e6c03a33b01d13a.jpg", "truck"],
    [f"{DATA_FOLDER}/383f5f3edc4dd305ba0d1eb57ac55892.jpg", "family"],
    [f"{DATA_FOLDER}/395e11b1214a0308f6a6e7834cebc379.jpg", "taxi"],
    [f"{DATA_FOLDER}/0419f656d61b42a6b2d567ed9ab6673f.jpg", "truck"],
    [f"{DATA_FOLDER}/433a57b6d7767b974279ef8fcd49cc16.jpg", "truck"],
    [f"{DATA_FOLDER}/508bc40d51bc31a77507595e528155e9.jpg", "taxi"],
    [f"{DATA_FOLDER}/766f0049a40eea217328d2975312d3e0.jpg", "truck"],
    [f"{DATA_FOLDER}/897c7be0d09ce1e992ab8e68fc3194f6.jpg", "racing"],
    [f"{DATA_FOLDER}/1570cbe2a81dd646e94bf5e4e3b1f2b1.jpg", "tuck"],
    [f"{DATA_FOLDER}/3037a9c243c43addf623e9609a09515c.jpg", "family"],
    [f"{DATA_FOLDER}/5789a84b08a72667f6ed1dc0e1400778.jpg", "jeep"],
    [f"{DATA_FOLDER}/6739b35e5a470fdb37862fb2bc5251ef.jpg", "bus"],
    [f"{DATA_FOLDER}/7141dc0b260cb428c6c0bcb5e51426fc.jpg", "truck"],
    [f"{DATA_FOLDER}/7855d33398a070b855381b5c0022a12a.jpg", "truck"],
    [f"{DATA_FOLDER}/7911da38a55c13ad65ee761999873d82.jpg", "truck"],
    [f"{DATA_FOLDER}/8189cbaa92299ea536be3ad23ddf1d00.jpg", "truck"],    
    [f"{DATA_FOLDER}/9129e0d08ecc8343ba0d45b297cb19d2.jpg", "jeep"],
    [f"{DATA_FOLDER}/9803fa77a01d28b1be669c4f2dc1a9d2.jpg", "family"],
    [f"{DATA_FOLDER}/45242c276cd516adb93561503a2f6ed7.jpg", "bus"],
    [f"{DATA_FOLDER}/62795dabbbbe425b28275c4a0a601a63.jpg", "truck"],
    [f"{DATA_FOLDER}/91409b182fe012ec4a4ffde63bf4f3e1.jpg", "jeep"],
    [f"{DATA_FOLDER}/99043f3239c13157c6f90db35fb7d412.jpg", "racing"],
    [f"{DATA_FOLDER}/6250695ca39f3d4e37303996a08a75e0.jpg", "truck"],
    [f"{DATA_FOLDER}/7678977975a2ab3fc51efbafd0baca37.jpg", "jeep"],
    [f"{DATA_FOLDER}/a2fec0c55e7c73d27fac134b9074846e.jpg", "taxi"],
    [f"{DATA_FOLDER}/a3c4f639c87e59383cfec1062b0ebd1b.jpg", "family"],
    [f"{DATA_FOLDER}/a3c9175c79d68d3a8aaf8719e59519d0.jpg", "jeep"],
    [f"{DATA_FOLDER}/a67cad73fd03d2d5cef3a81bf78ed0a3.jpg", "bus"],    
    [f"{DATA_FOLDER}/a72e4b986609cd50953f4c070f9df9a0.jpg", "jeep"],
    [f"{DATA_FOLDER}/aba61896c1391587795060868f00f6e9.jpg", "racing"],
    [f"{DATA_FOLDER}/adaf348564dd71ae2c318378d5f60051.jpg", "racing"],
    [f"{DATA_FOLDER}/ae648d1b856a00e9150e84a0af1c1e71.jpg", "truck"],
    [f"{DATA_FOLDER}/b0d111a7c355e47d040ba1f2282c04cb.jpg", "family"],
    [f"{DATA_FOLDER}/b52fbc88bc83acea1b0d374ccb460178.jpg", "taxi"],
    [f"{DATA_FOLDER}/b62c7d59cd74dc1bc2eaa1b951cfd0a7.jpg", "bus"],
    [f"{DATA_FOLDER}/b64c45c9bdffb475ad76641a8f111c49.jpg", "jeep"],
    [f"{DATA_FOLDER}/b0322b716be09e89ac21c22874194836.jpg", "truck"],
    [f"{DATA_FOLDER}/b00473a9f517bbfc5923c95288b40ff2.jpg", "taxi"],
    [f"{DATA_FOLDER}/be8a535960c82e4332eed118196f7d56.jpg", "jeep"],
    [f"{DATA_FOLDER}/c1adb1337eba00e584e3704af99d291c.jpg", "bus"],
    [f"{DATA_FOLDER}/c9ef804987778a3e0590e5e501f38b56.jpg", "family"],
    [f"{DATA_FOLDER}/ca68371ad50fa0575b1fac8c5796fbfb.jpg", "jeep"],
    [f"{DATA_FOLDER}/cd0e83983e36613ec5b154d2d6ddb0d1.jpg", "taxi"],
    [f"{DATA_FOLDER}/d0bc0f8b0754ea0f5c8736c769aaebaa.jpg", "truck"],
    [f"{DATA_FOLDER}/d4bb40dee633de3a610857f75d8fb037.jpg", "jeep"],
    [f"{DATA_FOLDER}/d6c4bc7127a99edbd4bcc615b771604d.jpg", "family"],
    [f"{DATA_FOLDER}/db1ada5a60a1562ac68263444a2595be.jpg", "jeep"],
    [f"{DATA_FOLDER}/ddfe9b3fd10abc7e11d7479ea32ec569.jpg", "jeep"],
    [f"{DATA_FOLDER}/def5ed8eda194f16fe75b17278d7dffd.jpg", "truck"],
    [f"{DATA_FOLDER}/e4d643544391d8c00db061075616e1b9.jpg", "racing"],
    [f"{DATA_FOLDER}/e8d28df25bb2b713af60360bd29e1ab4.jpg", "racing"],
    [f"{DATA_FOLDER}/e7299b9187f09e190127873fb0725e1c.jpg", "taxi"],
    [f"{DATA_FOLDER}/eafaa5fd38df370046f8b185488e7fa5.jpg", "bus"],
    [f"{DATA_FOLDER}/eb3e8701367c730dbdd6dc9298ebea61.jpg", "family"],
    [f"{DATA_FOLDER}/ebd1939d557a824d89e1017780d5a69c.jpg", "truck"],
    [f"{DATA_FOLDER}/f0a70d450423aa8c88d87a0a0c9c1acf.jpg", "family"],
    [f"{DATA_FOLDER}/f2cd6c805f084ce28812a62a0a582e7a.jpg", "family"],
    [f"{DATA_FOLDER}/f9d5513079dfa5dd43c3c996aeebd88b.jpg", "truck"],
    [f"{DATA_FOLDER}/f55500934c8cc648bc9ae7b39cecf6bc.jpg", "truck"],
    [f"{DATA_FOLDER}/f400677842675240319e69bc2e5998eb.jpg", "truck"],
    [f"{DATA_FOLDER}/fa218c0050ba94160ec0873ec80c4807.jpg", "family"],
    [f"{DATA_FOLDER}/fb1e25b446c8174cc443c366a1d0e38b.jpg", "bus"],
    [f"{DATA_FOLDER}/fe6ba3df4ec47d2107d4aed4844d0ca9.jpg", "racing"],
    [f"{DATA_FOLDER}/ff9974e9177a989100fae4b8c505cad5.jpg", "bus"]
]

# Carregar configuració
with open('vehicles_config.json', 'r') as f:
    config = json.load(f)

# Treu les dades de test del zip
def decompress_data_zip(type):
    # Esborra la carpeta de test
    if os.path.exists(DATA_FOLDER):
        shutil.rmtree(DATA_FOLDER)

    # Descomprimeix l'arxiu que conté les carpetes de test
    zip_filename = f"./data/{type}.zip"
    extract_to = './data/'
    with zipfile.ZipFile(zip_filename, 'r') as zipf:
        for member in zipf.namelist():
            # Filtra per ignorar carpetes ocultes i per extreure només la carpeta
            if member.startswith(f"{type}/") and not member.startswith('__MACOSX/'):
                zipf.extract(member, extract_to)

# Crea una nova instància del model ResNet18 sense pesos. 
# Aquesta configuració s'utilitza per construir el mateix 
# tipus de model que s'ha fet servir durant l'entrenament.
def create_model(num_classes):
    """Crear el mateix model que fem servir per entrenar"""
    # Crear una instància del model ResNet18
    model = resnet18(weights=None)
    
    # Obtenir el nombre de característiques d'entrada de la capa final
    num_ftrs = model.fc.in_features
    
    # Substituir la capa final amb la mateixa estructura que l'entrenament
    model.fc = nn.Sequential(
        nn.Dropout(config['model_params']['dropout_rate']),
        nn.Linear(num_ftrs, num_classes)  # Eliminat Sigmoid ja que usem CrossEntropy
    )
    
    return model

# Retorna un conjunt de transformacions d'imatge utilitzades durant el procés de validació. 
# Aquestes transformacions asseguren que les imatges que es passen al model tenen
# la mateixa estructura i escales que les imatges utilitzades durant l'entrenament
def get_transform():
    return transforms.Compose([
        transforms.Resize(tuple(config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['normalize_mean'], 
                           std=config['normalize_std'])
    ])

# Carrega un model de xarxa neuronal prèviament entrenat des d'un fitxer especificat
def load_model(model_path, num_classes, device):
    """Carregar el model amb els pesos entrenats"""
    model = create_model(num_classes)
    
    # Afegim weights_only=True per evitar el warning i millorar la seguretat
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model

# Avaluar el rendiment del model utilitzant un conjunt de dades de test. 
# L'objectiu és obtenir les prediccions per a cada imatge 
# del conjunt de test i comparar-les amb les etiquetes reals
def evaluate_model(model, test_images, transform, class_names, device):
    correct = 0
    total = 0
    predictions = []
    
    with torch.no_grad():
        for img_path, true_label in test_images:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            # Obtenim les probabilitats per cada classe
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            
            # Obtenim la classe amb major probabilitat
            _, predicted_idx = torch.max(outputs.data, 1)
            predicted_label = class_names[predicted_idx.item()]
            
            # Obtenim la confiança (probabilitat) de la predicció
            confidence = probabilities[predicted_idx].item()
            
            # Comprovar si la predicció és correcta
            is_correct = predicted_label == true_label
            if is_correct:
                correct += 1
            total += 1
            
            predictions.append({
                'image': Path(img_path).name,
                'predicted': predicted_label,
                'true_label': true_label,
                'confidence': confidence,
                'correct': is_correct
            })
            
            # Mostrem el resultat principal
            print(f"\nImage: {Path(img_path).name}, Prediction: {confidence:.2%} = {"'"+predicted_label+"'":9} ({"'"+true_label+"'":9} > {'correct' if is_correct else 'wrong'})")
            
            # Mostrem les probabilitats per cada classe
            probs_str = " | ".join([f"{class_names[i]}: {prob.item()*100:6.2f}%" for i, prob in enumerate(probabilities)])
            print(f"{probs_str}")
    
    return correct, total

def main():
    # Descomprimir les dades de test
    decompress_data_zip("testing")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type == "cuda" or device.type == "mps":
        print(f"Using device: {device} (GPU accelerated)")
    else:
        print(f"Using device: {device} (CPU based)")
    
    class_names = config['classes']
    print(f"Classes: {class_names}")
    
    # Carreguem el model amb el número correcte de classes
    model = load_model(config['model_path'], len(class_names), device)
    transform = get_transform()
    
    correct, total = evaluate_model(
        model, test_images, transform, class_names, device
    )
    
    accuracy = correct / total
    print("\nGlobal results:")
    print(f"Total images: {total}")
    print(f"Hits: {correct}")
    print(f"Accuracy: {accuracy:.2%}")

    if os.path.exists(DATA_FOLDER):
        shutil.rmtree(DATA_FOLDER)

if __name__ == "__main__":
    main()