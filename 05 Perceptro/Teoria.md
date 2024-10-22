<div style="display: flex; width: 100%;">
    <div style="flex: 1; padding: 0px;">
        <p>© Albert Palacios Jiménez, 2024</p>
    </div>
    <div style="flex: 1; padding: 0px; text-align: right;">
        <img src="./assets/ieti.png" height="32" alt="Logo de IETI" style="max-height: 32px;">
    </div>
</div>
<br/>

# Perceptró

El **perceptró** és un algorisme d'intel·ligència artificial que es pot imaginar com una màquina que pren decisions senzilles basades en dades. 

És una manera bàsica d'ensenyar a un programa a reconèixer patrons i classificar informació entre 0 i 1.

El funcionament dels perceptrons s'inspira en el funcionament de les neurones animals. A partir d'unes senyals d'entrada, poden activar o no la senyal de sortida cap a altres neurones:

<center>
<img src="./assets/neuron.png" style="max-width: 90%; width: 400px; max-height: 400px;" alt="">
<br/>
</center>
<br/>

Amb un sol perceptró només es poden classificar dades **linealment separables**, això vol dir dades que se poden classificar en un "Si" o en un "No".

- Temperatures fredes vs caluroses
- Correu normal o "spam"
- Punts que estàn per sobre o per sota d'una linia en un dibuix
- ...

Amb múltiples perceptrons es formen les **xarxes neurals**, i segons com s'organitzen els perceptrons ja es poden fer classificacions més complexes.

## Parts d'un perceptró:

**Entrades:**

- Imagina que les entrades són com les dades que rep el perceptró per prendre una decisió. Per exemple, podrien ser números binaris (0 i 1), com els que fem servir per representar dades en un ordinador.

    Les entrades són la informació que volem que el perceptró analitzi, com si fos una pregunta que li fem.

**Funció d'activació:**

- És la part que fa els càlculs. La funció d'activació combina totes les entrades i decideix si la màquina ha de "activar-se" o no.

    Es pot pensar com un interruptor: si rep suficient "energia" (les entrades tenen cert valor), s'encén; si no, es queda apagat.


    Aquesta "energia" es calcula sumant totes les entrades i comparant el resultat amb un llindar (un valor que decidim). Si la suma supera el llindar, el perceptró es "activa".

- Els **pesos** es poden imaginar com la importància que donem a cadascuna de les entrades.

    Si una entrada té un pes alt, significa que és més important per a la decisió final. 
    
    Si té un pes baix, significa que és menys important.
Per exemple, si el perceptró rep dues dades, A i B, i el pes d'A és més gran que el de B, llavors A tindrà més influència en la decisió del perceptró.
Biaix:

- El **biaix** és com un ajust que permet al perceptró prendre decisions més flexibles.

    És com establir una línia de partida diferent: encara que les entrades no tinguin molt pes, el biaix pot ajudar a "activar" el perceptró si és prou alt.

    Pots imaginar el biaix com un punt extra que fa més fàcil (o difícil) que el perceptró es posi en marxa. Sense el biaix, el perceptró només es "encendria" si les entrades són molt fortes.

La funció d'activació sempre és:

- Multiplicar cada entrada pel seu pes
- Sumar tots els resultats de les multiplicacions anteriors
- Sumar el *biaix*
- Comprovar si el resultat anterior és major o igual que 0

**Sortida:**

- Després de calcular les entrades, el perceptró dona una resposta: "0" o "1".
Podem imaginar-ho com un semàfor: si el perceptró es "activa", la llum es posa verda (sortida "1"); si no es "activa", es queda vermella (sortida "0").

    Això permet al perceptró classificar les dades en dos grups diferents: per exemple, números parells i senars.

    Amb aquesta estructura, el perceptró pot aprendre a distingir coses senzilles i és el fonament de molts sistemes més avançats d'intel·ligència artificial.

<br/>
<center>
<img src="./assets/perceptron.png" style="max-width: 90%; width: 400px; max-height: 400px;" alt="">
<br/>
</center>
<br/>
<center>
Entrades -> Pesos i Biaix -> Funció d'activació -> Sortida
</center>

## Fases:

Un cop hem dissenyat el nostre perceptró, segons les necessitats de classificació de dades que tenim, hem de procedir amb:

Fes anar l'exemple amb:

```bash
./run.sh com.project.Main
```

**Fase d'entrenament**: 

A partir d'unes entrades conegudes, s'entrena al perceptró perquè aprengui com classificar cada entrada.

- Durant l'entrenament, el perceptró revisa els exemples moltes vegades (això es diu "èpoques"). Això li permet millorar gradualment, com si estigués practicant una activitat per ser més bo.

- Per a cada exemple de l'entrenament, el perceptró intenta fer una predicció (és a dir, decidir si el número és parell o senar).

- Després, comprova si s'ha equivocat o no, comparant la seva resposta amb la resposta correcta.

- Si la seva predicció no és correcta, sabem que ha comès un error, i aquest error es fa servir per corregir els pesos i el biaix perquè la pròxima vegada pugui encertar.

**Fase de classificació**: 

Un cop entrenat el perceptró, se li donen noves entrades (que poden ser iguals o diferents a les de la fase d'entrenament), per veure com les classifica.

## Exemple 0

En aquest exemple veiem com es fa servir un perceptró per classificar números entre **"parell"** i **"senar"**

**Entrenament**: 

- L'algorisme ajusta els pesos i el biaix basant-se en la diferència (error) entre la sortida prevista i la correcta.

- Si la classificació és incorrecta, els pesos s'ajusten per minimitzar l'error en futures prediccions.

**Classificació**:

- Donada una entrada (número binari), el perceptró calcula una suma ponderada.

- Si la suma és positiva o zero, classifica com a "senar"; si és negativa, com a "parell".

```java
public class Perceptron {

    // Provar diferents EPOCHS: 5, 25, 50, 75 fins a 100
    private const int EPOCHS = 100; 

    private double[] weights;
    private double bias;
    private double learningRate = 0.1;

    public Perceptron(int inputSize) {
        weights = new double[inputSize];
        bias = 0.0;
        // Inicialització dels pesos aleatòriament
        for (int i = 0; i < weights.length; i++) {
            weights[i] = Math.random() - 0.5;
        }
    }

    private int activate(double sum) {
        return sum >= 0 ? 1 : 0; // Sortida 1 per senar, 0 per parell
    }

    private int predict(int[] inputs) {
        double sum = bias;
        for (int i = 0; i < weights.length; i++) {
            sum += weights[i] * inputs[i];
        }
        return activate(sum);
    }

    public void train(int[][] inputData, int[] labels, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < inputData.length; i++) {
                int prediction = predict(inputData[i]);
                int error = labels[i] - prediction;

                // Ajustar pesos i bias
                for (int j = 0; j < weights.length; j++) {
                    weights[j] += learningRate * error * inputData[i][j];
                }
                bias += learningRate * error;
            }
        }
    }

    public double testAccuracy(int[][] inputData, int[] labels) {
        int correct = 0;
        for (int i = 0; i < inputData.length; i++) {
            int prediction = predict(inputData[i]);
            boolean isCorrect = (prediction == labels[i]);
            if (isCorrect) {
                correct++;
            }

            // Imprimir entrada i resultat
            System.out.print("Entrada: ");
            for (int bit : inputData[i]) {
                System.out.print(bit);
            }
            System.out.println(" -> Resultat: " + (prediction == 0 ? "Parell" : "Senar") +
                               " (Esperat: " + (labels[i] == 0 ? "Parell" : "Senar") + ")");
        }
        return (correct / (double) inputData.length) * 100.0;
    }

    public static void main(String[] args) {

        Perceptron perceptron = new Perceptron(4);

        // Entrades en binari per a 4 bits (0000 = 0, 0001 = 1, etc.)
        int[][] inputs = {
            {0, 0, 0, 0}, // 0 (parell)
            {0, 0, 0, 1}, // 1 (senar)
            {0, 0, 1, 0}, // 2 (parell)
            {0, 0, 1, 1}, // 3 (senar)
            {0, 1, 0, 0}, // 4 (parell)
            {0, 1, 0, 1}, // 5 (senar)
            {0, 1, 1, 0}, // 6 (parell)
            {0, 1, 1, 1}, // 7 (senar)
        };

        // Etiquetes de sortida (0 per parell, 1 per senar)
        int[] labels = {0, 1, 0, 1, 0, 1, 0, 1};

        // Entrenar el perceptró
        perceptron.train(inputs, labels, EPOCHS);

        // Calcular el % d'encert amb totes les entrades
        double accuracy = perceptron.testAccuracy(inputs, labels);
        System.out.println("Percentatge d'encert: " + accuracy + "%");
    }
}
```