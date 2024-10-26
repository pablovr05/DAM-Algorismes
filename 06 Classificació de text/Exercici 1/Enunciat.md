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
... resultats previs ...

Text: @JetBlue Of course U know I would like 2 lay you d..., Prediction:  65.82% = "positive" ("positive" > correct)
Text: @VirginAmerica heyyyy guyyyys.. been trying to get..., Prediction:  97.45% = "negative" ("negative" > correct)
Text: @united with the purchase of my ticket i am entitl..., Prediction:  94.62% = "negative" ("negative" > correct)
Text: @united @gg8929 Ladies and gents - United Airlines..., Prediction:  61.59% = "neutral" ("neutral" > correct)
Text: Welp. “@JetBlue: Our fleet's on fleek. http://t.co..., Prediction:  82.76% = "neutral" ("neutral" > correct)
Text: @VirginAmerica Applied for Status Match on Feb 1. ..., Prediction:  91.78% = "negative" ("negative" > correct)

Validation of 50 examples: success: 43/50, accuracy: 86.00, Error rate: 0.14
```