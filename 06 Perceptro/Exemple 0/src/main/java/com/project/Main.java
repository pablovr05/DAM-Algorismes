package com.project;

import java.util.*;  

public class Main {

    // Provar diferents EPOCHS: 1, 2, ... 10
    private static final int EPOCHS = 10; 

    private double[] weights;
    private double bias;
    private double learningRate = 0.1;

    public Main(int inputSize) {
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

        Main perceptron = new Main(4);

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