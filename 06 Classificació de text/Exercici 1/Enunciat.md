# Exercici 1

Cal entrenar una xarxa neuronal que permeti classificar textos d'opinió sobre una aerolinia en: 

- negative, neutral, positive

Fes servir l'arxiu './data/airline.csv'

Fixa't que les columnes que tenen informació important per fer la classificació són:

- airline_sentiment, text

## Tasques:

0) Fes els arxius **"model_config.json"** i **"ai_train.py"** per entrenar la xarxa anterior i generar els arxius **"model_metadata.json"** i **"model_network.pth"**

1) Fes un arxiu **"ai_classify.py"** que esculli 50 textos a l'arzar de l'arxiu **"./data/ariline.csv"** i mostri les estadistiques de classificar-los amb la xarxa de l'arpartat 0

2) Fes un arxiu **"ai_classify_single.py"** que demana per input: "What's your opinion about the airline?" i fa servir la xarxa anterior per dir: "Your opinion about the airline is X" on X pot ser 'negative, neutral, positive'

3) Fes un document **"millores.pdf"** en el que expliquins quines configuracions es poden posar a la xarxa per millorar els resultats obtinguts.

<br/><br/>

**Nota**: La classificació de la tasca 1 ha de ser de l'estil:

```text
... 49 resultats previs ...

Text: @AmericanAir Can't unload flight #3322 because jet..., Prediction: 99.82% = "negative" ("negative" > correct)
negat: 1.00 ,neutr: 0.00 ,posit: 0.00

Global results:
  Total texts: 50
  Hits: 46
  Accuracy: 92.00%
```