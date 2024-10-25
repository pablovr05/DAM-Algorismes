<div style="display: flex; width: 100%;">
    <div style="flex: 1; padding: 0px;">
        <p>© Albert Palacios Jiménez, 2024</p>
    </div>
    <div style="flex: 1; padding: 0px; text-align: right;">
        <img src="./assets/ieti.png" height="32" alt="Logo de IETI" style="max-height: 32px;">
    </div>
</div>
<br/>

# Introducció a la IA

## Xarxes Neuronals

Les **xarxes neuronals** estàn inspirades en el funcionament del cervell animal, i permeten automatitzar tasques de classificació i reconeixement de patrons.

Les **xarxes neuronals** estàn compostes per unitats anomenades **perceptrons** que representen les neurones, i es connecten entre si a través de pesos que determinen la influència d'una neurona sobre una altra.

<center>
<img src="./assets/networklayers.png" style="max-width: 90%; width: 400px; max-height: 400px;" alt="">
<br/>
</center>
<br/>

Les **xarxes neuronals** es composen de:

- Una capa d'entrada (input)
- Una o diverses capes ocultes (hidden)
- Una capa de sortida (output)

Cada una de les capes que formen la xarxa, pot tenir una o diverses neurones.

Hi ha molts tipus de xarxes, segons la seva funció:

<center>
<img src="./assets/typesofnetworksslice.png" style="max-width: 90%; width: 400px; max-height: 400px;" alt="">
<br/>
</center>
<br/>

[Imatge amb tipus de xarxes neurals](./assets/typesofnetworks.webp)

Quan volem fer tasques de programació amb xarxes neurals, hem de definir quines capes tindrà la nostra xarxa. 

Per les tasques més comuns ja hi ha xarxes estàndard, com per exemple:

- **ResNet18** que és un tipus de xarxa preparada per classificar imatges
- **BERT** que és un tipus de xarxa preparada per treballar amb text
- **GPT** capaç de generar text
- ...

### PyTorch

PyTorch és una biblioteca de codi obert per a l'aprenentatge automàtic i la computació científica, que proporciona eines per construir i entrenar xarxes neuronals.

<center>
<img src="./assets/logo-pytorch.png" style="max-width: 90%; width: 200px; max-height: 400px;" alt="">
<br/>
</center>
<br/>


**PyTorch** és coneguda per la seva facilitat d'ús, flexibilitat i suport per a càlculs en GPU, permetent així desenvolupar models complexos de manera eficient.

Per instal·lar **PyTorch**:
```bash
pip install torch torchvision torchaudio Pillow tqdm pandas scikit-learn numpy transformers
```

O bé a macOS amb *brew*:
```bash
python3 -m pip install torch torchvision torchaudio Pillow tqdm pandas scikit-learn numpy transformers --break-system-package
```

Com que **PyTorch** pot fer servir l'acceleració gràfica, cal llibreries extra per activar-la:
```bash
# NVIDIA
# Cuda: https://developer.nvidia.com/cuda-downloads
# cuDNN: https://developer.nvidia.com/cudnn
pip install torch torchvision torchaudio Pillow tqdm --extra-index-url https://download.pytorch.org/whl/cu118
# ARM
# https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html
pip install torch torchvision torchaudio Pillow tqdm --extra-index-url https://download.pytorch.org/whl/rocm5.4.2
```

Podeu provar si teniu les llibreries PyTorch i l'acceleració a través de GPU activada amb l'script Python:

```bash
./ai_utils_check.py
# O bé, python ./ai_utils_check.py
```

<br/><br/>

# Classificació de textos
