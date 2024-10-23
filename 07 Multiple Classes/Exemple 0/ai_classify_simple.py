#!/usr/bin/env python3

# python3 -m pip install torch torchvision torchaudio Pillow tqdm --break-system-package

import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

model_path = 'vehicle_model.pth'
types_path = 'vehicle_types.json'

# Imatges de test amb les seves etiquetes
test_images = [
    ["./data/test/0abad4440c4415b9707f11151762526c.jpg", "racing"],
    ["./data/test/0ea9451a8de9cf7581835ccb20940174.jpg", "family"],
    ["./data/test/1a3f888928097b855e49c1926b94a5dc.jpg", "truck"],
    ["./data/test/1b97666eaae94e70cba5e238017cab7a.jpg", "truck"],
    ["./data/test/1ded9b835b61b963f13d6b4374ef2492.jpg", "jeep"],
    ["./data/test/2b6da117b6b37e8fa1d48cddec05b5b2.jpg", "racing"],
    ["./data/test/2b96ccd24db32c2c47581afde8d42722.jpg", "bus"],
    ["./data/test/2f313143d05ede99c2458ba697161c1c.jpg", "truck"],
    ["./data/test/3acbdbd78d3270f626fa9eac8c182bb3.jpg", "family"],
    ["./data/test/3db10fa5700a75911c64410f35d7b94d.jpg", "racing"],
    ["./data/test/3dbb4fcf817656a9a865f3ea33f07a6a.jpg", "racing"],
    ["./data/test/3f7a8b1ae0bc3d5d82e1240f810ce47d.jpg", "truck"],
    ["./data/test/4aa43e160ee667c2deb922cd371204bc.jpg", "jeep"],
    ["./data/test/4b71b5a2fa94a71b0581c6650a0f8d0a.jpg", "truck"],
    ["./data/test/4baa8146de477ebafae4b3f46d5614c2.jpg", "truck"],
    ["./data/test/05a006ebd8667a39988a253b10956bb7.jpg", "racing"],
    ["./data/test/5d7514af404266c3f92efcee8662f640.jpg", "bus"],
    ["./data/test/5edeade7ca9e85efa46e1be268be129c.jpg", "truck"],
    ["./data/test/6a4e8af47c5b5b1412ce9a28a247d50e.jpg", "family"],
    ["./data/test/6f13f51f98bd1e2332bfa6f1c1c5ff14.jpg", "jeep"],
    ["./data/test/7a9027a09a9149fd901f97630a052b29.jpg", "taxi"],
    ["./data/test/7fe25c5c90c4f447fe3fa10adf4232cd.jpg", "bus"],
    ["./data/test/09bbbd3f758e129a2692844e3b62d5a8.jpg", "family"],
    ["./data/test/9cecedb8bfe67dbbc76d7723c3b4de43.jpg", "truck"],
    ["./data/test/9e37ae8bf11600a30afcd592d73a4be1.jpg", "bus"],
    ["./data/test/34d68f0a8eda8aadc568c1fd14b222b9.jpg", "family"],
    ["./data/test/37d329f08d105885c3e5d658921356c6.jpg", "family"],
    ["./data/test/48dc3c3c0038b87e330d7692cae4b0ff.jpg", "family"],
    ["./data/test/51a610fa467c9b6d0a0c9ebc87ebd6cd.jpg", "truck"],
    ["./data/test/61b29def40ee024890ebc2af8851a120.jpg", "truck"],
    ["./data/test/63f9f7dc8298317284bf7dbf2074b3ba.jpg", "bus"],
    ["./data/test/75f2660d690d5bc035c1f9d9424e253a.jpg", "truck"],
    ["./data/test/81eba54e555030c678afe71bb81c3480.jpg", "jeep"],
    ["./data/test/84c8aadbabbeb753042593faffa4e376.jpg", "truck"],
    ["./data/test/95ac3b0debba83ee706b4edef03ee2c2.jpg", "truck"],
    ["./data/test/98b83106dcaeeb585913fb59c8525757.jpg", "family"],
    ["./data/test/210d3833b9c3b0613e6c03a33b01d13a.jpg", "truck"],
    ["./data/test/383f5f3edc4dd305ba0d1eb57ac55892.jpg", "family"],
    ["./data/test/395e11b1214a0308f6a6e7834cebc379.jpg", "taxi"],
    ["./data/test/0419f656d61b42a6b2d567ed9ab6673f.jpg", "truck"],
    ["./data/test/433a57b6d7767b974279ef8fcd49cc16.jpg", "truck"],
    ["./data/test/508bc40d51bc31a77507595e528155e9.jpg", "taxi"],
    ["./data/test/766f0049a40eea217328d2975312d3e0.jpg", "truck"],
    ["./data/test/897c7be0d09ce1e992ab8e68fc3194f6.jpg", "racing"],
    ["./data/test/1570cbe2a81dd646e94bf5e4e3b1f2b1.jpg", "tuck"],
    ["./data/test/3037a9c243c43addf623e9609a09515c.jpg", "family"],
    ["./data/test/5789a84b08a72667f6ed1dc0e1400778.jpg", "jeep"],
    ["./data/test/6739b35e5a470fdb37862fb2bc5251ef.jpg", "bus"],
    ["./data/test/7141dc0b260cb428c6c0bcb5e51426fc.jpg", "truck"],
    ["./data/test/7855d33398a070b855381b5c0022a12a.jpg", "truck"],
    ["./data/test/7911da38a55c13ad65ee761999873d82.jpg", "truck"],
    ["./data/test/8189cbaa92299ea536be3ad23ddf1d00.jpg", "truck"],    
    ["./data/test/9129e0d08ecc8343ba0d45b297cb19d2.jpg", "jeep"],
    ["./data/test/9803fa77a01d28b1be669c4f2dc1a9d2.jpg", "family"],
    ["./data/test/45242c276cd516adb93561503a2f6ed7.jpg", "bus"],
    ["./data/test/62795dabbbbe425b28275c4a0a601a63.jpg", "truck"],
    ["./data/test/91409b182fe012ec4a4ffde63bf4f3e1.jpg", "jeep"],
    ["./data/test/99043f3239c13157c6f90db35fb7d412.jpg", "racing"],
    ["./data/test/6250695ca39f3d4e37303996a08a75e0.jpg", "truck"],
    ["./data/test/7678977975a2ab3fc51efbafd0baca37.jpg", "jeep"],
    ["./data/test/a2fec0c55e7c73d27fac134b9074846e.jpg", "taxi"],
    ["./data/test/a3c4f639c87e59383cfec1062b0ebd1b.jpg", "family"],
    ["./data/test/a3c9175c79d68d3a8aaf8719e59519d0.jpg", "jeep"],
    ["./data/test/a67cad73fd03d2d5cef3a81bf78ed0a3.jpg", "bus"],    
    ["./data/test/a72e4b986609cd50953f4c070f9df9a0.jpg", "jeep"],
    ["./data/test/aba61896c1391587795060868f00f6e9.jpg", "racing"],
    ["./data/test/adaf348564dd71ae2c318378d5f60051.jpg", "racing"],
    ["./data/test/ae648d1b856a00e9150e84a0af1c1e71.jpg", "truck"],
    ["./data/test/b0d111a7c355e47d040ba1f2282c04cb.jpg", "family"],
    ["./data/test/b52fbc88bc83acea1b0d374ccb460178.jpg", "taxi"],
    ["./data/test/b62c7d59cd74dc1bc2eaa1b951cfd0a7.jpg", "bus"],
    ["./data/test/b64c45c9bdffb475ad76641a8f111c49.jpg", "jeep"],
    ["./data/test/b0322b716be09e89ac21c22874194836.jpg", "truck"],
    ["./data/test/b00473a9f517bbfc5923c95288b40ff2.jpg", "taxi"],
    ["./data/test/be8a535960c82e4332eed118196f7d56.jpg", "jeep"],
    ["./data/test/c1adb1337eba00e584e3704af99d291c.jpg", "bus"],
    ["./data/test/c9ef804987778a3e0590e5e501f38b56.jpg", "family"],
    ["./data/test/ca68371ad50fa0575b1fac8c5796fbfb.jpg", "jeep"],
    ["./data/test/cd0e83983e36613ec5b154d2d6ddb0d1.jpg", "taxi"],
    ["./data/test/d0bc0f8b0754ea0f5c8736c769aaebaa.jpg", "truck"],
    ["./data/test/d4bb40dee633de3a610857f75d8fb037.jpg", "jeep"],
    ["./data/test/d6c4bc7127a99edbd4bcc615b771604d.jpg", "family"],
    ["./data/test/db1ada5a60a1562ac68263444a2595be.jpg", "jeep"],
    ["./data/test/ddfe9b3fd10abc7e11d7479ea32ec569.jpg", "jeep"],
    ["./data/test/def5ed8eda194f16fe75b17278d7dffd.jpg", "truck"],
    ["./data/test/e4d643544391d8c00db061075616e1b9.jpg", "racing"],
    ["./data/test/e8d28df25bb2b713af60360bd29e1ab4.jpg", "racing"],
    ["./data/test/e7299b9187f09e190127873fb0725e1c.jpg", "taxi"],
    ["./data/test/eafaa5fd38df370046f8b185488e7fa5.jpg", "bus"],
    ["./data/test/eb3e8701367c730dbdd6dc9298ebea61.jpg", "family"],
    ["./data/test/ebd1939d557a824d89e1017780d5a69c.jpg", "truck"],
    ["./data/test/f0a70d450423aa8c88d87a0a0c9c1acf.jpg", "family"],
    ["./data/test/f2cd6c805f084ce28812a62a0a582e7a.jpg", "family"],
    ["./data/test/f9d5513079dfa5dd43c3c996aeebd88b.jpg", "truck"],
    ["./data/test/f55500934c8cc648bc9ae7b39cecf6bc.jpg", "truck"],
    ["./data/test/f400677842675240319e69bc2e5998eb.jpg", "truck"],
    ["./data/test/fa218c0050ba94160ec0873ec80c4807.jpg", "family"],
    ["./data/test/fb1e25b446c8174cc443c366a1d0e38b.jpg", "bus"],
    ["./data/test/fe6ba3df4ec47d2107d4aed4844d0ca9.jpg", "racing"],
    ["./data/test/ff9974e9177a989100fae4b8c505cad5.jpg", "bus"]
]

print("Define transformations")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Carregar les etiquetes des de l'arxiu JSON
with open(types_path, 'r') as f:
    class_names = json.load(f)

num_classes = len(class_names)  # Nombre de classes basat en el JSON

print("Define AI model")
model = resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)  # Ajusta a la quantitat de classes

# Carregar els pesos del model entrenat
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
if device.type == "cuda" or device.type == "mps":
    print(f"Using device: {device} (GPU accelerated)")
else:
    print(f"Using device: {device} (CPU based)")

print("Start classification")
predictions = {}

with torch.no_grad():
    for img_path, label in test_images:
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_label = predicted.item()

        predicted_class = class_names[predicted_label]
        match = predicted_class == label

        print(f"Image: {img_path}, Predicted: {"'"+predicted_class+"'":8} ({"'"+label+"'":8} : {match})")

print("Classification completed.")