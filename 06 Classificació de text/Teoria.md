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

[Imatge amb tipus de xarxes neuronals](./assets/typesofnetworks.webp)

Quan volem fer tasques de programació amb xarxes neuronals, hem de definir quines capes tindrà la nostra xarxa. 

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
pip install torch torchvision torchaudio Pillow tqdm pandas numpy scikit-learn transformers
```

O bé a macOS amb *brew*:
```bash
python3 -m pip install torch torchvision torchaudio Pillow tqdm pandas numpy scikit-learn transformers --break-system-package
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

<br/>

# Paràmetres de les xarxes neuronals

Quan configurem una xarxa neuronal hem de tenir en compte:

**Dades:** On està la font de informació que fem servir per entrenar la xarxa

**Etiquetes:** Si la informació està etiquetada, on trobem aquestes etiquetes i com les relacionem amb cada test.

- En arxius de text (normalment **".csv"**) les etiquetes estàn en alguna de les columnes de l'axiu, i el text a classificar en una altre

- En arxius d'imatge, les etiquetes pot ser el nom de les carpetes on es guarda cada tipus d'imatge.

**Model IA:** Quan es configura la xarxa, s'han de determinar dades importants, com quantes capes té, quantes neurones té cada capa, quin tipus de funció fa servir la neurona per activar-se, quin tipus de sortida dona (binaria, múltiple), ...

**Entrenament:** Cal definir els paràmetres d'entrenament, per exemple els EPOCHS

**Optimització:** Cal definir els paràmetres de la optimització

    Al optimitzar es pot intentar millorar la xarxa, respecte la millor versió que hem trobat, unes quantes vegades. Però amb un limit, aquest limit es diu paciència.

## Entrenament

Entrenar la xarxa consisteix en passar-li dades que ja tenim etiquetades, i ajustar els pesos en conseqüència.

Aquest procés el fa unes quantes vegades, de cada intent se'n diu EPOCH

**validació-creuada**

La validació creuada guarda una part de les dades d'entrenament per validar si l'eficàcia de l'últim entrenament.

Així l'entrenament queda dividit en dues fases:

- **train**: que és pròpiament l'entrenament
- **evaluate**: que és la validació de l'últim entrenament

Si la validació ens diu que l'últim entrenament ha millorat la xarxa, respecte la millor versió dels entrenaments anteriors, aleshores guardem la xarxa com a la millor versió aconseguida.

Si la validació ens diu que l'últim entrenament no ha millorat la xarxa, respecte la millor versió que tenim, aleshores la descartem i la tornem a entrenar.

**early-stopping**

Quan definim la configuració establim el número d'**epochs** que s'han d'intentar per entrenar la xarxa. És a dir quans cops l'hem d'entrenar i comparar amb els entrenaments anteriors.

A vegades, anar fent el cicle de **epochs** no fa que trobem una xarxa millor, i només perdem el temps.

Si veiem que la xarxa no millora podem aturar l'entrenament. És a dir, acabar l'entrenament abans del previst (**early-stopping**)

Un dels mètodes de **early-stopping** és definint la **patience**, que són els cops que intentem seguir entrenant la xarxa sense que millori, abans d'aturar l'entrenament.


# Classificació de textos

Una de les principals tasques de les IAs és classificar textos.

Hi ha molts tipus de classificació, aquí en veurem dos exemples:

- **Classificació binària**: classificar un text com a pertanyent a un grup o no. 

    Exemples: dir si un correu és spam, si un anunci és de cuina, si una proposta té males intencions, ...

- **Classificació multi class**: classificar un text amb una categoria, a partir d'un conjunt d'etiquetes o categories.

    Exemple: classificar una noticia com a (politica, esports, cultura, ...)

## Arxius .csv

És habitual que les fonts d'informació que entrenen les xarxes neuronals siguin arxius **".csv"**

Els arxius **".csv"** tenen la informació separada per comes *(",")*, de fet:

```text
CSV = Comma Separated Values
```

La primera fila dels arxius **".csv"** té les etiquetes del què significa cada columna.

Les següents files tenen la informació en format de taula, però amb cada camp separat per comes *(",")*

```csv
sentiment,comentari
positiu,"M'encanta aquest producte, és fantàstic!"
negatiu,És el pitjor que he comprat mai.
positiu,"Servei absolutament increïble, molt content!"
```

## Configuració "model_config.json"

Quan es configura la xarxa neuronal per ser entrenada, cal indicar:

```json
    "paths": {
        "data": "./data/spam.csv",
        "trained_network": "model_network.pth",
        "metadata": "model_metadata.json"
    },
    "columns": {
        "categories": "label",
        "text": "mail"
    },
```

- **data**: L'arxiu que conté les dades **".csv"**

- **columns/catgories**: El nom de la columna de l'arxiu **".csv"** que classifica la categoria del text

- **columns/text**: El nom de la columna de l'arxiu **".csv"** on hi ha el text a classificar

A l'arxi de configuració de la xarxa **"model_config.json"** hi ha altres paràmetres importants:

- **is_binary**: Si es tracta d'unca classificació binària o de múltiples categories

- **layers**: Com està formada la xarxa neuronal

- **training/epochs**: El número màxim d'entrenaments que es faràn

- **optimization/early_stopping/patience**: La paciència abans d'aturar l'entrenament si no està millorant

## Entrenament 'ai_train.py'

L'arxiu **"ai_train.py"** entrena una xarxa segons la configuració de **"model_config.json"**

Durant l'entrenament es mostra informació de cada EPOCH d'aquest estil:

```text

Training:   100.00% |████████████████████| 366/366 [00:07<00:00, 47.07it/s]
Evaluating: 100.00% |████████████████████| 92/92 [00:01<00:00, 75.53it/s]
Epoch 1/32 (Train: loss: 0.84, accuracy: 0.63) (Eval: loss: 0.70, accuracy: 0.74)
New best model saved with eval accuracy 73.53%
```

Es pot veure com en cada iteració **EPOCH** es fa un entrenament i una valoració d'aquest entrenament

Si la nova xarxa entrenada és millor que la que teníem, es guarda com a **model** vàlid.

## Arxius resultants de l'entrenament

Un cop s'ha entrenat la xarxa es guarda el model entrenat i la informació associada al model als arxius definits per **paths/trained_network** i **metadata**. Per defecte són:

- **model_network.pth** el millor model aconseguit durant l'entrenament

- **metadata** les dades associades al model, per poder fer-lo servir

## Classificació 'ai_classify.py'

L'arxiu **"ai_classify.py"** fa servir la xarxa entrenada, per classificar 50 textos a l'atzar del conjunt de dades original.

A partir d'aquí, mira quants cops s'ha classificat correctament i genera estadístiques sobre el funcionament del model:

```text
... altres textos abaluats ...

Text: You made my day. Do have a great day too.         ..., Prediction:  99.74% = "ham"  ("ham"  > correct)
Text: I was just callin to say hi. Take care bruv!      ..., Prediction:  99.89% = "ham"  ("ham"  > correct)
Text: Id onluy matters when getting on from offcampus   ..., Prediction:  98.97% = "ham"  ("ham"  > correct)
Text: Ok... Let u noe when i leave my house.            ..., Prediction:  99.99% = "ham"  ("ham"  > correct)

Validation of 50 examples: success: 50/50, accuracy: 100.00, Error rate: 0.00
```

# Exemple 0 (ham or spam)

L'**SPAM** és un embotit molt popular al Regne Unit, ja que acompanya sovint els esmorzars.

<center>
<img src="./assets/spam.png" style="max-width: 90%; width: 250px; max-height: 400px;" alt="">
<br/>
</center>
<br/>

És un producte que els clients no solen demanar, i és un substitut barat del pernil.

Per aquest motiu, el correu escombraria va agafar el sobrenom de **SPAM**. Perquè el reps encara que no el demanis.

A l'exemple 0 s'entrena una **xarxa neuronal** per classificar textos de correus entre:

- **Ham** (pernil), correu bò
- **Spam**, correu basura

És per tant, un entrenament **binari**

Cal fixar-se que la configuració **"model-config.json"** defineix les columnes de l'arxiu **"spam.csv"**:

- **label** columna que té l'etiqueta del text (ham o spam)
- **mail** columna que té el text a classificar

<br/>

# Exemple 1 (categories de notícies)

En aquest exemple s'entrena una **xarxa neuronal** per classificar notícies segons categories.

És per tant, un entrenament **multi class** perquè s'ecull una classe d'un conjunt de categories.

Cal fixar-se que la configuració **"model-config.json"** defineix les columnes de l'arxiu **"news.csv"**:

- **category** columna que té l'etiqueta del text (ham o spam)
- **body** columna que té el text a classificar