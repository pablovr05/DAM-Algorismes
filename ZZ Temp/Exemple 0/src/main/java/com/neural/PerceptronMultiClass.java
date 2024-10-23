package com.neural;

import java.util.*;

public class PerceptronMultiClass extends Perceptron {
    private double[][] weights;
    private double[] biases;
    private double learningRate;
    private int numClasses;

    public PerceptronMultiClass(int inputSize, int numClasses, double learningRate) {
        this.learningRate = learningRate;
        this.numClasses = numClasses;
        this.weights = new double[numClasses][inputSize];
        this.biases = new double[numClasses];
        for (int i = 0; i < numClasses; i++) {
            for (int j = 0; j < inputSize; j++) {
                weights[i][j] = Math.random() - 0.5;
            }
            biases[i] = 0.0;
        }
    }

    public int getInputSize() {
        return weights[0].length;
    }

    public int getNumClasses() {
        return numClasses;
    }

    public double getLearningRate() {
        return learningRate;
    }

    @Override
    public double[] predict(int[] inputs) {
        double[] outputs = new double[numClasses];
        for (int i = 0; i < numClasses; i++) {
            double sum = biases[i];
            for (int j = 0; j < inputs.length; j++) {
                sum += weights[i][j] * inputs[j];
            }
            outputs[i] = sum;
        }
        return softmax(outputs); // Assume softmax returns double[]
    }

    private double[] softmax(double[] inputs) {
        double max = Arrays.stream(inputs).max().getAsDouble();
        double sum = 0.0;
        double[] expValues = new double[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            expValues[i] = Math.exp(inputs[i] - max);
            sum += expValues[i];
        }
        for (int i = 0; i < inputs.length; i++) {
            expValues[i] /= sum;
        }
        return expValues;
    }

    public void train(int[][] inputData, int[] labels, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < inputData.length; i++) {
                double[] outputs = new double[numClasses];
                for (int k = 0; k < numClasses; k++) {
                    double sum = biases[k];
                    for (int j = 0; j < inputData[i].length; j++) {
                        sum += weights[k][j] * inputData[i][j];
                    }
                    outputs[k] = sum;
                }

                // Convertir a probabilitats amb Softmax
                double[] probabilities = softmax(outputs);

                // Codificar l'etiqueta correcta com "one-hot"
                double[] target = new double[numClasses];
                target[labels[i]] = 1.0;

                // Ajustar pesos i biases
                for (int k = 0; k < numClasses; k++) {
                    double error = target[k] - probabilities[k];
                    for (int j = 0; j < inputData[i].length; j++) {
                        weights[k][j] += learningRate * error * inputData[i][j];
                    }
                    biases[k] += learningRate * error;
                }
            }
        }
    }

    @Override
    public double testAccuracy(int[][] inputData, int[] labels) {
        int correct = 0;
        for (int i = 0; i < inputData.length; i++) {
            double[] prediction = predict(inputData[i]);
            
            // Find the class with the highest probability or score
            int predictedClass = 0;
            for (int j = 1; j < prediction.length; j++) {
                if (prediction[j] > prediction[predictedClass]) {
                    predictedClass = j;
                }
            }
    
            boolean isCorrect = (predictedClass == labels[i]);
            if (isCorrect) {
                correct++;
            }
    
            System.out.print("Input: ");
            for (int bit : inputData[i]) {
                System.out.print(bit);
            }
            System.out.println(" -> Result: " + predictedClass + " (Expected: " + labels[i] + ")");
        }
        return (correct / (double) inputData.length) * 100.0;
    }
}