package com.project;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class Main {

    // Provar diferents EPOCHS: 1, 2, ... 1000
    private static final int EPOCHS = 10; 

    public static void main(String[] args) {

        // Configuració de la xarxa: Perceptró amb una sola capa oculta
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

        // Provar una nova entrada per predir si és parell o senar
        INDArray testInput = Nd4j.create(new double[]{0, 1, 1, 0}).reshape(1, 4); 
        INDArray outputPrediction = model.output(testInput);

        double result = outputPrediction.getDouble(0);
        System.out.println("Resultat de la predicció (0=parell, 1=senar): " + result);
        System.out.println(result < 0.5 ? "Parell" : "Senar");

        // Calcular l'encert amb les dades d'entrenament
        double accuracy = calculateAccuracy(model, dataSet);
        System.out.println("Percentatge d'encert: " + accuracy + "%");
    }

    // Mètode per calcular l'encert del model amb el conjunt de dades donades
    private static double calculateAccuracy(MultiLayerNetwork model, DataSet dataSet) {
        int correct = 0;
        INDArray input = dataSet.getFeatures();
        INDArray labels = dataSet.getLabels();
        
        for (int i = 0; i < input.rows(); i++) {
                INDArray row = input.getRow(i).reshape(1, 4);
                double actual = labels.getDouble(i);
                double prediction = model.output(row).getDouble(0);

                // Convertir l'entrada binària a decimal
                int decimalValue = (int) (row.getDouble(0) * 8 + row.getDouble(1) * 4 + row.getDouble(2) * 2 + row.getDouble(3) * 1);

                boolean isCorrect = (prediction < 0.5 && actual == 0) || (prediction >= 0.5 && actual == 1);

                // Mostrar predicció amb valor decimal i resultat esperat
                System.out.print("Entrada: " + row + " (" + decimalValue + ") -> Predicció: " + (prediction < 0.5 ? "Parell" : "Senar"));
                System.out.println(" (Esperat: " + (actual == 0 ? "Parell" : "Senar") + ")");

                if (isCorrect) {
                        correct++;
                }
        }
        return (correct / (double) input.rows()) * 100.0;
    }   
}
