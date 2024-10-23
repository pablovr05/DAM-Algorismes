package com.neural;

import java.util.*;

class PerceptronMultiLabel extends Perceptron {
    private double[][] weights;
    private double[] biases;
    private double learningRate;
    private int numLabels;

    public PerceptronMultiLabel(int inputSize, int numLabels, double learningRate) {
        this.learningRate = learningRate;
        this.numLabels = numLabels;
        this.weights = new double[numLabels][inputSize];
        this.biases = new double[numLabels];
        for (int i = 0; i < numLabels; i++) {
            for (int j = 0; j < inputSize; j++) {
                weights[i][j] = Math.random() - 0.5;
            }
            biases[i] = 0.0;
        }
    }

    public int getInputSize() {
        return weights[0].length;
    }

    public int getNumLabels() {
        return numLabels;
    }

    public double getLearningRate() {
        return learningRate;
    }

    @Override
    public double[] predict(int[] inputs) {
        double[] outputs = new double[numLabels];
        for (int i = 0; i < numLabels; i++) {
            double sum = biases[i];
            for (int j = 0; j < inputs.length; j++) {
                sum += weights[i][j] * inputs[j];
            }
            outputs[i] = sigmoid(sum); // Assume sigmoid returns a double
        }
        return outputs;
    }

    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public void train(int[][] inputData, int[][] labels, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < inputData.length; i++) {
                double[] outputs = predict(inputData[i]);

                // Ajustar pesos i biases
                for (int k = 0; k < numLabels; k++) {
                    double error = labels[i][k] - outputs[k];
                    for (int j = 0; j < inputData[i].length; j++) {
                        weights[k][j] += learningRate * error * inputData[i][j];
                    }
                    biases[k] += learningRate * error;
                }
            }
        }
    }
   
    @Override
    public double testAccuracy(int[][] inputData, int[][] labels) {
        int correct = 0;
        for (int i = 0; i < inputData.length; i++) {
            double[] predictions = predict(inputData[i]);

            // Decidir si el color està present (0.5 com a llindar)
            boolean isCorrect = true;
            System.out.print("Input: ");
            for (int bit : inputData[i]) {
                System.out.print(bit);
            }
            System.out.print(" -> Predicted Colors: ");
            for (int j = 0; j < predictions.length; j++) {
                int predictedLabel = predictions[j] >= 0.5 ? 1 : 0;
                System.out.print(predictedLabel + " ");
                if (predictedLabel != labels[i][j]) {
                    isCorrect = false;
                }
            }
            System.out.println(isCorrect ? "(Correct)" : "(Incorrect)");
            if (isCorrect) {
                correct++;
            }
        }
        return (correct / (double) inputData.length) * 100.0;
    }
}