package com.project;

import java.util.*;  

public class Main {
    // Provar per diferents EPOCHS: 1, 2, ... 10
    private static final int EPOCHS = 10;

    public static void main(String[] args) {

        // Crea una instància d'un perceptró
        // inputSize = 4 que son els bits 0/1 de l'entrada
        // activationType = "relu" que és fer una suma simple de cada entrada pel seu pes
        // learningRate = 0.1 que defineix el canvi dels pesos durant l'entrenament, en cas d'error/encert
        Perceptron perceptron = new Perceptron(4, "relu", 0.1);

        // Valors d'entrada possibles (4 bits)
        int[][] inputs = {
            {0, 0, 0, 0}, // Num 0
            {0, 0, 0, 1}, // Num 1
            {0, 0, 1, 0}, // Num 2
            {0, 0, 1, 1}, // Num 3
            {0, 1, 0, 0}, // Num 4
            {0, 1, 0, 1}, // Num 5
            {0, 1, 1, 0}, // Num 6
            {0, 1, 1, 1}  // Num 7
        };

        // Als labels 0 és parell, 1 imparell
        int[] labels = {
            0, // Num 0 -> label 0 parell 
            1, // Num 1 -> label 1 imparell
            0, // Num 2 -> label 0 parell
            1, // Num 3 -> label 1 imparell
            0, // Num 4 -> label 0 parell
            1, // Num 5 -> label 1 imparell
            0, // Num 6 -> label 0 parell
            1  // Num 7 -> label 1 imparell
        };

        // Fes l'entrenament del perceptró
        perceptron.train(inputs, labels, EPOCHS);
        
        // Medeix la precissió del perceptró
        double accuracy = perceptron.testAccuracy(inputs, labels);
        System.out.println("\nPercentatge d'encert del Perceptró entrenat amb " + EPOCHS + " EPOCHS: " + accuracy + "%\n");
    }
}