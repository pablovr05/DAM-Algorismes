package com.neural;

public abstract class Perceptron {
    
    public abstract int getInputSize();
    public abstract double getLearningRate();

    public String getActivationType() {
        throw new UnsupportedOperationException("getActivationType() not supported for this perceptron type.");
    }

    public int getNumLabels() {
        throw new UnsupportedOperationException("getNumLabels() not supported for this perceptron type.");
    }

    public void train(int[][] inputData, int[] labels, int epochs) {
        throw new UnsupportedOperationException("int[][] inputData, int[] labels, int epochs) not supported for this perceptron type.");
    }

    public void train(int[][] inputData, int[][] labels, int epochs) {
        throw new UnsupportedOperationException("int[][] inputData, int[][] labels, int epochs) not supported for this perceptron type.");
    }

    public double testAccuracy(int[][] inputData, int[] labels) {
        throw new UnsupportedOperationException("testAccuracy(int[][], int[]) not supported for this perceptron type.");
    }

    public double testAccuracy(int[][] inputData, int[][] labels) {
        throw new UnsupportedOperationException("testAccuracy(int[][] inputData, int[][] labels) not supported for this perceptron type.");
    }
  
    protected abstract double[] predict(int[] inputs);
}

