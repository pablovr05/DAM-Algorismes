package com.neural;

import java.util.List;
import java.util.ArrayList;
import java.io.FileWriter;
import java.io.IOException;
import org.json.JSONArray;
import org.json.JSONObject;

public class NeuralNetwork {
    private List<Layer> layers;

    public NeuralNetwork() {
        layers = new ArrayList<>();
    }

    // Method to add a layer of perceptrons
    public void addLayer(Class<? extends Perceptron> perceptronType, int count, int inputSize, String activationType, double learningRate, int numClassesOrLabels) {
        Layer layer = new Layer(perceptronType, count, inputSize, activationType, learningRate, numClassesOrLabels);
        layers.add(layer);
    }

    public void train(int[][] inputData, Object labels, int epochs) {
        for (Layer layer : layers) {
            layer.train(inputData, labels, epochs);
        }
    }

    public double testAccuracy(int[][] inputData, Object labels) {
        double totalAccuracy = 0;
        for (Layer layer : layers) {
            totalAccuracy += layer.testAccuracy(inputData, labels);
        }
        return totalAccuracy / layers.size();
    }

    // Save layers information to JSON
    public void saveToJson(String filePath) {
        JSONArray layersArray = new JSONArray();
        for (Layer layer : layers) {
            JSONObject layerObject = new JSONObject();
            for (Perceptron perceptron : layer.getPerceptrons()) {
                JSONObject perceptronData = new JSONObject();
                if (perceptron instanceof PerceptronBinary) {
                    PerceptronBinary binary = (PerceptronBinary) perceptron;
                    perceptronData.put("type", "binary");
                    perceptronData.put("inputSize", binary.getInputSize());
                    perceptronData.put("activationType", binary.getActivationType());
                    perceptronData.put("learningRate", binary.getLearningRate());
                } else if (perceptron instanceof PerceptronMultiClass) {
                    PerceptronMultiClass multiClass = (PerceptronMultiClass) perceptron;
                    perceptronData.put("type", "multi-class");
                    perceptronData.put("inputSize", multiClass.getInputSize());
                    perceptronData.put("numClasses", multiClass.getNumClasses());
                    perceptronData.put("learningRate", multiClass.getLearningRate());
                } else if (perceptron instanceof PerceptronMultiLabel) {
                    PerceptronMultiLabel multiLabel = (PerceptronMultiLabel) perceptron;
                    perceptronData.put("type", "multi-label");
                    perceptronData.put("inputSize", multiLabel.getInputSize());
                    perceptronData.put("numLabels", multiLabel.getNumLabels());
                    perceptronData.put("learningRate", multiLabel.getLearningRate());
                }
                layerObject.append("perceptrons", perceptronData);
            }
            layersArray.put(layerObject);
        }

        JSONObject neuralNetworkJson = new JSONObject();
        neuralNetworkJson.put("layers", layersArray);

        try (FileWriter file = new FileWriter(filePath)) {
            file.write(neuralNetworkJson.toString(4));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
