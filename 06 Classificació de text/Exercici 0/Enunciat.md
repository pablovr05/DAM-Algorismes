# Exercici 0

Cal entrenar una xarxa neuronal que permeti classificar si un text en anglès o no, segons les categories: 

- eng, other

Fes servir l'arxiu './data/engorother.csv'

Fixa't que les columnes que tenen informació important per fer la classificació són:

- label, phrase

## Tasques:

0) Fes els arxius **"model_config.json"** i **"ai_train.py"** per entrenar la xarxa anterior i generar els arxius **"model_metadata.json"** i **"model_network.pth"**

1) Fes un arxiu **"ai_classify.py"** que esculli 50 textos a l'arzar de l'arxiu **"./data/sentiments.csv"** i mostri les estadistiques de classificar-los amb la xarxa de l'arpartat 0

2) Fes un arxiu **"ai_classify_single.py"** que demana per input: "Write something ..." i fa servir la xarxa anterior per dir si s'ha escrit en anglès o en un altre idioma. Segons el resultat:

    - Si és **eng** mostra. "This is English"
    - Si és **other** mostra. "I don't understand you"

3) Fes un document **"millores.pdf"** en el que expliquins quines configuracions es poden posar a la xarxa per millorar els resultats obtinguts.

<br/><br/>

**Nota**: La classificació de la tasca 1 ha de ser de l'estil:

```text
... resultats previs ...

Text: What time do you usually have dinner?             ..., Prediction:  91.79% = "eng"  ("eng"  > correct)
Text: Qual è la password del WiFi?                      ..., Prediction:  91.74% = "other" ("other" > correct)
Text: The lights are not working                        ..., Prediction:  86.22% = "eng"  ("eng"  > correct)
Text: La batería está muerta                            ..., Prediction:  96.78% = "other" ("other" > correct)
Text: L'ascenseur est en panne                          ..., Prediction:  86.23% = "other" ("other" > correct)
Text: I need to schedule an appointment                 ..., Prediction:  94.27% = "eng"  ("eng"  > correct)

Validation of 50 examples: success: 50/50, accuracy: 100.00, Error rate: 0.00
```