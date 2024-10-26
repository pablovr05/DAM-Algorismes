# Exercici 0

Cal entrenar una xarxa neuronal que permeti classificar si un text mostra un sentiment positiu o negatiu, segons les categories: 

- sentiment,comentari

Fes servir l'arxiu './data/sentiments.csv'

Fixa't que les columnes que tenen informació important per fer la classificació són:

- category, text

## Tasques:

0) Fes els arxius **"model_config.json"** i **"ai_train.py"** per entrenar la xarxa anterior i generar els arxius **"model_metadata.json"** i **"model_network.pth"**

1) Fes un arxiu **"ai_classify.py"** que esculli 50 textos a l'arzar de l'arxiu **"./data/sentiments.csv"** i mostri les estadistiques de classificar-los amb la xarxa de l'arpartat 0

2) Fes un arxiu **"ai_classify_single.py"** que demana per input: "Write something?" i fa servir la xarxa anterior per dir si és 'positiu' o 'negatiu'. Segons el resultat:

    - Si és **positiu** mostra. "Sentiment positiu"
    - Si és **negatiu** mostra. "Sentiment negatiu"

3) Fes un document **"millores.pdf"** en el que expliquins quines configuracions es poden posar a la xarxa per millorar els resultats obtinguts.

<br/><br/>

**Nota**: La classificació de la tasca 1 ha de ser de l'estil:

```text
... resultats previs ...

Text: Em penedeixo d'haver-lo comprat, totalment inútil...., Prediction:  69.71% = "negatiu" ("negatiu" > correct)
Text: El recomanaria a altres sense dubtar-ho!          ..., Prediction:  76.44% = "positiu" ("positiu" > correct)
Text: Mai vaig rebre la meva comanda, experiència horrib..., Prediction:  78.52% = "negatiu" ("negatiu" > correct)
Text: Sembla barat i fràgil, molt insatisfet.           ..., Prediction:  59.94% = "negatiu" ("negatiu" > correct)
Text: La nova versió és excel·lent, estic emocionat!    ..., Prediction:  77.23% = "positiu" ("positiu" > correct)

Validation of 50 examples: success: 50/50, accuracy: 100.00, Error rate: 0.00
```