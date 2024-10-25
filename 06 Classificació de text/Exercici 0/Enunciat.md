# Exercici 0

Cal entrenar una xarxa neuronal que permeti classificar si un text és en anglès o en una altra llengua, segons les categories: 

- eng, other

Fes servir l'arxiu './data/engorother.csv'

Fixa't que les columnes que tenen informació important per fer la classificació són:

- category, text

## Tasques:

0) Fes els arxius **"model_config.json"** i **"ai_train.py"** per entrenar la xarxa anterior i generar els arxius **"model_metadata.json"** i **"model_network.pth"**

1) Fes un arxiu **"ai_classify.py"** que esculli 50 textos a l'arzar de l'arxiu **"./data/ariline.csv"** i mostri les estadistiques de classificar-los amb la xarxa de l'arpartat 0

2) Fes un arxiu **"ai_classify_single.py"** que demana per input: "Write something?" i fa servir la xarxa anterior per dir si és 'eng' o 'other'. Segons el resultat:

    - Si és **eng** respon. "This is english"
    - Si és **other** respon. "I don't understand you"

3) Fes un document **"millores.pdf"** en el que expliquins quines configuracions es poden posar a la xarxa per millorar els resultats obtinguts.

<br/><br/>

**Nota**: La classificació de la tasca 1 ha de ser de l'estil:

```text
... 48 resultats previs ...

Text: Ich bin ein Berliner.                             ..., Prediction: 100.00% = "eng"  ("other" > wrong)
eng: 1.00

Text: As in a mirror...                                 ..., Prediction: 100.00% = "eng"  ("eng"  > correct)
eng: 1.00

Global results:
  Total texts: 50
  Hits: 20
  Accuracy: 40.00%
```