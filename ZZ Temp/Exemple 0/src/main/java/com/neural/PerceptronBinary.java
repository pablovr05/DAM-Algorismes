package com.neural;

import java.util.*;

public class PerceptronBinary extends Perceptron {
    private double[] weights;
    private double bias;
    private double learningRate;
    private ActivationFunction activationFunction;

    public PerceptronBinary(int inputSize, String activationType, double learningRate) {
        this.learningRate = learningRate;
        this.activationFunction = ActivationFunction.getActivation(activationType);
        weights = new double[inputSize];
        bias = 0.0;
        for (int i = 0; i < weights.length; i++) {
            weights[i] = Math.random() - 0.5;
        }
    }

    public int getInputSize() {
        return weights.length;
    }

    public String getActivationType() {
        return activationFunction.getType();
    }

    public double getLearningRate() {
        return learningRate;
    }

    @Override
    protected double[] predict(int[] inputs) {
        double sum = bias;
        for (int i = 0; i < weights.length; i++) {
            sum += weights[i] * inputs[i];
        }
        return new double[]{activationFunction.activate(sum)};
    }

    @Override
    public void train(int[][] inputData, int[] labels, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < inputData.length; i++) {
                double prediction = predict(inputData[i])[0];
                double error = labels[i] - prediction;

                for (int j = 0; j < weights.length; j++) {
                    weights[j] += learningRate * error * inputData[i][j];
                }
                bias += learningRate * error;
            }
        }
    }

    @Override
    public double testAccuracy(int[][] inputData, int[] labels) {
        int correct = 0;
        for (int i = 0; i < inputData.length; i++) {
            double[] predictionArray = predict(inputData[i]);
            int prediction = predictionArray[0] >= 0.5 ? 1 : 0; // Convert double to binary (0 or 1)
            if (prediction == labels[i]) {
                correct++;
            }
        }
        return (correct / (double) inputData.length) * 100.0;
    }
}
