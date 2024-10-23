package com.project;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;

import com.neural.NeuralNetwork;
import com.neural.PerceptronBinary;

public class Main {
    private static final int EPOCHS = 10;
    private static final int INPUT_SIZE = 128 * 128; // Mida de les imatges
    
    private static final String TRAIN_IMAGES_PATH = "data/train";
    private static final String LABELS_CSV_PATH = "data/labels.csv";
    private static final String[] VALID_COLORS = {
        "white", "gray", "black", "red", "green", "blue", "yellow", "orange", "purple", "pink"
    };
    private static final Map<String, Integer> COLOR_MAP = createColorMap();

    public static void main(String[] args) {
        NeuralNetwork neuralNetwork = new NeuralNetwork();
        
        // Replace addLayer with addPerceptron, adding a PerceptronBinary instance
        PerceptronBinary binaryPerceptron = new PerceptronBinary(INPUT_SIZE, "sigmoid", 0.1);
        neuralNetwork.addPerceptron(binaryPerceptron);

        Map<String, int[]> labelsMap = loadLabels(LABELS_CSV_PATH);
        List<int[]> inputsList = new ArrayList<>();
        List<int[]> labelsList = new ArrayList<>();

        loadInputDataAndMatchLabels(TRAIN_IMAGES_PATH, labelsMap, inputsList, labelsList);

        int[][] inputs = inputsList.toArray(new int[0][]);
        int[] labels = convertMultiLabelToSingle(labelsList); // Convert to single-label format

        // Train with single-label format data
        neuralNetwork.train(inputs, labels, EPOCHS);
        double accuracy = neuralNetwork.testAccuracy(inputs, labels);
        System.out.println("Accuracy: " + accuracy + "%");

        neuralNetwork.saveToJson("trained_model.json");
    }

    // Convert multi-label format to single-label format for binary perceptron
    private static int[] convertMultiLabelToSingle(List<int[]> labelsList) {
        int[] singleLabels = new int[labelsList.size()];
        for (int i = 0; i < labelsList.size(); i++) {
            int[] labelArray = labelsList.get(i);
            
            // Convert multi-label to binary (Assume label is 1 if any color is present)
            singleLabels[i] = containsColor(labelArray) ? 1 : 0;
        }
        return singleLabels;
    }

    // Helper method to determine if a label contains any color
    private static boolean containsColor(int[] labelArray) {
        for (int value : labelArray) {
            if (value == 1) {
                return true;
            }
        }
        return false;
    }

    private static void loadInputDataAndMatchLabels(String path, Map<String, int[]> labelsMap, List<int[]> inputsList, List<int[]> labelsList) {
        File folder = new File(path);
        File[] imageFiles = folder.listFiles((dir, name) -> name.toLowerCase().endsWith(".png"));

        if (imageFiles != null) {
            for (File imgFile : imageFiles) {
                try {
                    BufferedImage img = ImageIO.read(imgFile);
                    int[] binaryPixels = processImage(img);
                    int[] label = labelsMap.getOrDefault(imgFile.getName(), new int[VALID_COLORS.length]);
                    
                    inputsList.add(binaryPixels);
                    labelsList.add(label);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    private static Map<String, int[]> loadLabels(String csvPath) {
        Map<String, int[]> labelsMap = new HashMap<>();
        try (BufferedReader br = new BufferedReader(new FileReader(csvPath))) {
            String line;
            br.readLine(); // Saltar la capçalera
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                int[] labelVector = new int[VALID_COLORS.length];
                int bgIndex = COLOR_MAP.getOrDefault(parts[3].toLowerCase(), -1); // bg_color
                int fgIndex = COLOR_MAP.getOrDefault(parts[4].toLowerCase(), -1); // fg_color
                if (bgIndex != -1) labelVector[bgIndex] = 1;
                if (fgIndex != -1) labelVector[fgIndex] = 1;
                labelsMap.put(parts[1], labelVector); // filename com a clau
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return labelsMap;
    }

    private static int[] processImage(BufferedImage img) {
        int width = img.getWidth();
        int height = img.getHeight();
        int[] binaryPixels = new int[width * height];

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int pixel = img.getRGB(x, y);
                int gray = (pixel >> 16) & 0xff; // Treure la component vermella per a escala de grisos
                binaryPixels[y * width + x] = gray > 128 ? 1 : 0; // Binaritzar
            }
        }
        return binaryPixels;
    }

    private static Map<String, Integer> createColorMap() {
        Map<String, Integer> colorMap = new HashMap<>();
        for (int i = 0; i < VALID_COLORS.length; i++) {
            colorMap.put(VALID_COLORS[i], i);
        }
        return colorMap;
    }
}
