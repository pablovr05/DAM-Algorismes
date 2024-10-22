package com.train;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;

public class Main {

    private static final int EPOCHS = 10; // Èpoques d'entrenament
    private static final int WIDTH = 64;  // Mida de les imatges entrenades 64x64
    private static final int HEIGHT = 64; 
    private static final int CHANNELS = 3; // Imatges entrenades amb tons de gris (per imatges en color seria 3)
    private static final int BATCH_SIZE = 32; // Agrupar les imatges en lots (batches) de 32 abans d'ajustar pesos
    private static final int NUM_CLASSES = 2; // Categories que s'han de classificar (dues vol dir 'smile' i 'non_smile')
    
    public static void main(String[] args) throws Exception {
    
        // Crea un objecte File que apunta a la carpeta d'entrenament
        File trainData = new File("data/train");
    
        // Divideix els fitxers de la carpeta segons formats d'imatge permesos
        FileSplit trainSplit = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS);
    
        // Genera etiquetes de sortida basant-se en el nom de la carpeta que conté cada imatge ('smile' i 'non_smile')
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
    
        // Crea un lector d'imatges amb dimensions definides i el generador d'etiquetes
        ImageRecordReader trainRecordReader = new ImageRecordReader(HEIGHT, WIDTH, CHANNELS, labelMaker);
    
        // Inicialitza el lector d'imatges amb el conjunt de fitxers dividits
        trainRecordReader.initialize(trainSplit);
    
        // Crea un iterador de dades per carregar les imatges en lots per entrenament
        DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRecordReader, BATCH_SIZE, 1, NUM_CLASSES);
    
        // Estandaritza les imatges, per poder processar-les
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(trainIter);
        trainIter.setPreProcessor(scaler);
    
        // Configuració de la xarxa convolucional
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            // Defineix una llavor per a la inicialització aleatòria dels pesos per garantir resultats reproduïbles
            .seed(123)
            //Algorisme d'aprenentatge 'train' (Adam amb un ritme de 0.001)
            .updater(new Adam(0.001))
            // Comença la configuració de la llista de capes de la xarxa
            .list()
            // Afegeix una capa convolucional amb filtres de 5x5
            .layer(new ConvolutionLayer.Builder(5, 5)
                .nIn(CHANNELS) // Defineix el nombre de canals d'entrada (1 per tons de gris, 3 per RGB)
                .stride(1, 1)  // Estableix el desplaçament de la convolució a 1 (sense saltar píxels)
                .nOut(32)      // Defineix 32 filtres, cada un generant una sortida diferent
                .activation(Activation.RELU) // Utilitza ReLU com a funció d'activació
                .build())
            // Afegeix una capa de subsampling amb tipus de pooling màxim (max-pooling)
            .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2) // Utilitza una finestra de 2x2 per al pooling
                .stride(2, 2) // Desplaçament de 2, redueix la mida de les sortides a la meitat
                .build())
            // Afegeix una segona capa convolucional amb filtres de 5x5
            .layer(new ConvolutionLayer.Builder(5, 5)
                .nOut(64) // Defineix 64 filtres per a aquesta capa
                .stride(1, 1) // Estableix el desplaçament de la convolució a 1
                .activation(Activation.RELU) // Utilitza ReLU com a funció d'activació
                .build())
            // Afegeix una altra capa de subsampling amb max-pooling
            .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2) // Finestra de 2x2 per al pooling
                .stride(2, 2) // Desplaçament de 2 per reduir la mida
                .build())
            // Afegeix una capa densa totalment connectada amb 128 neurones
            .layer(new DenseLayer.Builder()
                .nOut(128) // Defineix 128 neurones en aquesta capa
                .activation(Activation.RELU) // Utilitza ReLU com a funció d'activació
                .build())
            // Afegeix la capa de sortida per a classificació
            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(NUM_CLASSES) // Defineix el nombre de classes de sortida (per exemple, 2 per binària)
                .activation(Activation.SOFTMAX) // Utilitza Softmax per a classificació multi-classe
                .build())
            // Estableix el tipus d'entrada esperat per la xarxa (convolucional amb altura, amplada, canals)
            .setInputType(org.deeplearning4j.nn.conf.inputs.InputType.convolutional(HEIGHT, WIDTH, CHANNELS))
            // Construeix la configuració completa de la xarxa
            .build();
    
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));
    
        // Entrenar el model
        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            System.out.println("Epoch " + (epoch + 1) + " / " + EPOCHS);
            model.fit(trainIter);
        }

        // Guardar el model entrenat
        model.save(new File("trained_model.zip"), true);
        System.out.println("Model entrenat i guardat com a 'trained_model.zip'");
    }
}
