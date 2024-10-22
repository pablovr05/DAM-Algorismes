<div style="display: flex; width: 100%;">
    <div style="flex: 1; padding: 0px;">
        <p>© Albert Palacios Jiménez, 2024</p>
    </div>
    <div style="flex: 1; padding: 0px; text-align: right;">
        <img src="./assets/ieti.png" height="32" alt="Logo de IETI" style="max-height: 32px;">
    </div>
</div>
<br/>

# Xarxes neuronals

Les **xarxes neuronals** és un sistema informàtic inspirat en el cervell humà. Així com el cervell està format per milions de neurones que treballen juntes per processar informació, una xarxa neuronal té moltes unitats petites anomenades **perceptrons** que treballen junts per resoldre problemes.

Les **xarxes neuronals** s'organitzen en capes:

- **Capa d'entrada**: és la primera capa i rep les dades inicials (com una imatge, text o nombres).

- **Capes ocultes (o denses)**: aquestes capes processen la informació. Cada neurona de la capa oculta pren les dades de la capa anterior, fa alguns càlculs, i les passa a la següent capa. Com més capes ocultes té una xarxa, més complexos són els patrons que pot reconèixer.

- **Capa de sortida**: és l'última capa que dona el resultat final (per exemple, dir si una imatge mostra un gos o un gat).

<center>
<img src="./assets/neuralnetwork.png" style="max-width: 90%; width: 400px; max-height: 400px;" alt="">
<br/>
</center>
<br/>

## Entrenament:

Les xarxes neuronals aprenen veient molts exemples.

Si volem que una xarxa aprengui a reconèixer gossos, li ensenyarem moltes fotos de gossos i li direm: "això és un gos". Amb cada exemple, la xarxa ajusta els seus càlculs per millorar.

Aquest procés d'ajust es diu **entrenament**, i es basa en corregir els errors que la xarxa fa fins que aprèn a donar la resposta correcta.

## Funcions d'activació:

Les **funcions d'activació** defineixen com s'activen les neurones a partir de l'entrada rebuda.

En un **perceptró la funció d'activació és lineal** perquè només s'activa si les dades rebudes són >=0.

Però segons el tipus de dades que analitzem necessitem altres tipus de funcions:

- **Funció Lineal (Identitat)**

    Simplement retorna el valor d'entrada tal com és.

    Útil en perceptrons simples i algunes capes de sortida.

- **Sigmoide**

    Converteix qualsevol valor en un rang entre 0 i 1, ideal per a sortides binàries.

- **ReLU (Rectified Linear Unit)**

    Retorna 0 si l'entrada és negativa, i el mateix valor si és positiu. Això permet que la xarxa aprengui amb més eficiència en tasques complexes.

    És molt popular en xarxes profundes.

- **Tangència Hiperbòlica (tanh)**

    Similar a la sigmoide però amb una sortida entre -1 i 1. 
    
    Pot ser útil quan les dades tenen valors negatius i positius.

<br/>
<center>
<img src="./assets/functions.png" style="max-width: 90%; width: 500px; " alt="">
<br/>
</center>
<br/>

Aquest exemple online permet veure l'efecte de les diferents funcions d'activació:

- [Tensofrflow playground](https://playground.tensorflow.org/)

<center>
<img src="./assets/playground.png" style="max-width: 90%; width: 500px; max-height: 400px;" alt="">
<br/>
</center>
<br/>

## Llibreria DL4J

La llibreria **deeplearnin4j** permet configurar xarxes neurals de manera senzilla amb Java.

## Exemple 0

Aquest exemple configura una d'xarxa d'una sola capa oculta amb dos perceptrons, un per l'entrada i un altre per la sortida fent servir **DL4J**.

Fes anar l'exemple amb:
```bash
./run.sh com.Project.Main
```

Codi:

```java
MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
    // Per iniciar els pesos 'weights' aleatòriament
    .seed(123) 
    // Algorisme d'aprenentatge 'train' (Sgd = Stochastic Gradient Descent)
    .updater(new Sgd(0.1))  
    // Activar diverses capes amb 'list' (per poder fer múltiples 'layers')
    .list()
    .layer(0, new DenseLayer.Builder()
        .nIn(4)  // 4 entrades
        .nOut(1) // 1 neurona de sortida
        .activation(Activation.SIGMOID) // funció d'activació
        .build())
    .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
        .nIn(1)  // 1 entrada de la capa anterior
        .nOut(1) // 1 sortida: 0 per parell, 1 per senar
        .activation(Activation.SIGMOID) // funció d'activació
        .build())
    // Construir una xarxa segons la configuració anterior
    .build();

// Inicialitzar la xarxa (segons config)
MultiLayerNetwork model = new MultiLayerNetwork(config);
model.init();
}
```

Entrena el model a partir d'un conjunt de dades d'entrenament. Dades d'entrada i les sortides esperades per aquestes entrades:

```java
// Entrades en binari per a 4 bits (0000 = 0, 0001 = 1, etc.)
INDArray input = Nd4j.create(new double[][]{
    {0, 0, 0, 0}, // 0 (parell)
    {0, 0, 0, 1}, // 1 (senar)
    {0, 0, 1, 0}, // 2 (parell)
    {0, 0, 1, 1}, // 3 (senar)
    {0, 1, 0, 0}, // 4 (parell)
    {0, 1, 0, 1}, // 5 (senar)
    {0, 1, 1, 0}, // 6 (parell)
    {0, 1, 1, 1}  // 7 (senar)
});

// Sortides corresponents a les entrades anteriors (0 per parell, 1 per senar)
INDArray output = Nd4j.create(new double[][]{
    {0}, // Parell
    {1}, // Senar
    {0}, // Parell
    {1}, // Senar
    {0}, // Parell
    {1}, // Senar
    {0}, // Parell
    {1}  // Senar
});

// Dades amb les que treballa la xarxa (entrada/sortida)
DataSet dataSet = new DataSet(input, output);

// Entrenar el model
for (int epoch = 0; epoch < EPOCHS; epoch++) {
    model.fit(dataSet);
}
```

Finalment executa la xarxa entrenada, enviant peticions i observant la resposta:
```java
INDArray testInput = Nd4j.create(new double[]{0, 1, 1, 0}).reshape(1, 4); 
INDArray outputPrediction = model.output(testInput);

double result = outputPrediction.getDouble(0);
System.out.println("Resultat de la predicció (0=parell, 1=senar): " + result);
System.out.println(result < 0.5 ? "Parell" : "Senar");
```

## Exemple 1

Aquest exemple configura una d'xarxa per classificar imatges de persones somrient (smile) o sense somriure (non_smile) amb **DL4J**.

Entrena la xarxa amb:

```bash
./run.sh com.train.Main
```

Classifica imatges:

```bash
./run.sh com.classify.Main
```