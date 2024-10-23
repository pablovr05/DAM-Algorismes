package com.neural;

import java.util.ArrayList;
import java.util.List;

public class Layer {
    private List<Perceptron> perceptrons;

    public Layer(Class<? extends Perceptron> perceptronType, int count, int inputSize, String activationType, double learningRate, int numClassesOrLabels) {
        perceptrons = new ArrayList<>();

        for (int i = 0; i < count; i++) {
            try {
                if (perceptronType == PerceptronBinary.class) {
                    perceptrons.add(new PerceptronBinary(inputSize, activationType, learningRate));
                } else if (perceptronType == PerceptronMultiClass.class) {
                    perceptrons.add(new PerceptronMultiClass(inputSize, numClassesOrLabels, learningRate));
                } else if (perceptronType == PerceptronMultiLabel.class) {
                    perceptrons.add(new PerceptronMultiLabel(inputSize, numClassesOrLabels, learningRate));
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    public List<Perceptron> getPerceptrons() {
        return perceptrons;
    }

    public void train(int[][] inputData, Object labels, int epochs) {
        for (Perceptron perceptron : perceptrons) {
            if (perceptron instanceof PerceptronBinary) {
                if (labels instanceof int[]) {
                    perceptron.train(inputData, (int[]) labels, epochs);
                } else {
                    throw new IllegalArgumentException("PerceptronBinary requires single-label data (int[]).");
                }
            } else if (perceptron instanceof PerceptronMultiClass) {
                if (labels instanceof int[]) {
                    perceptron.train(inputData, (int[]) labels, epochs);
                } else {
                    throw new IllegalArgumentException("PerceptronMultiClass requires single-label data (int[]).");
                }
            } else if (perceptron instanceof PerceptronMultiLabel) {
                if (labels instanceof int[][]) {
                    perceptron.train(inputData, (int[][]) labels, epochs);
                } else {
                    throw new IllegalArgumentException("PerceptronMultiLabel requires multi-label data (int[][]).");
                }
            }
        }
    }

    public double testAccuracy(int[][] inputData, Object labels) {
        double totalAccuracy = 0;
        for (Perceptron perceptron : perceptrons) {
            if (perceptron instanceof PerceptronBinary) {
                if (labels instanceof int[]) {
                    totalAccuracy += perceptron.testAccuracy(inputData, (int[]) labels);
                } else {
                    throw new IllegalArgumentException("PerceptronBinary requires single-label data (int[]).");
                }
            } else if (perceptron instanceof PerceptronMultiClass) {
                if (labels instanceof int[]) {
                    totalAccuracy += perceptron.testAccuracy(inputData, (int[]) labels);
                } else {
                    throw new IllegalArgumentException("PerceptronMultiClass requires single-label data (int[]).");
                }
            } else if (perceptron instanceof PerceptronMultiLabel) {
                if (labels instanceof int[][]) {
                    totalAccuracy += perceptron.testAccuracy(inputData, (int[][]) labels);
                } else {
                    throw new IllegalArgumentException("PerceptronMultiLabel requires multi-label data (int[][]).");
                }
            }
        }
        return totalAccuracy / perceptrons.size();
    }
}