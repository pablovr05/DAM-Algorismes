package com.project;

import java.util.*;

class Perceptron {
    private double[] weights;
    private double bias;
    private double learningRate;
    private ActivationFunction activationFunction;

    public Perceptron(int inputSize, String activationType, double learningRate) {
        this.learningRate = learningRate;
        this.activationFunction = ActivationFunction.getActivation(activationType);
        weights = new double[inputSize];
        bias = 0.0;
        for (int i = 0; i < weights.length; i++) {
            weights[i] = Math.random() - 0.5;
        }
    }

    private int predict(int[] inputs) {
        double sum = bias;
        for (int i = 0; i < weights.length; i++) {
            sum += weights[i] * inputs[i];
        }
        return activationFunction.activate(sum);
    }

    public void train(int[][] inputData, int[] labels, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < inputData.length; i++) {
                int prediction = predict(inputData[i]);
                int error = labels[i] - prediction;

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
            System.out.print("Input: ");
            for (int bit : inputData[i]) {
                System.out.print(bit);
            }
            System.out.println(" -> Result: " + prediction + " ("+ (labels[i] == prediction ? "ok" : "ko") + ")");
        }
        return (correct / (double) inputData.length) * 100.0;
    }   
}