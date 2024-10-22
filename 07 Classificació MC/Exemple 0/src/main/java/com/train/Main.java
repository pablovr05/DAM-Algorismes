package com.train;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.*;
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
import org.nd4j.common.primitives.Pair;
import java.util.Arrays;
import java.util.List;


import java.io.File;

public class Main {

    private static final int EPOCHS = 32;
    private static final int WIDTH = 128;
    private static final int HEIGHT = 128;
    private static final int CHANNELS = 1;
    private static final int BATCH_SIZE = 32;
    private static final int NUM_CLASSES = 10;

    public static void main(String[] args) throws Exception {

        // Directori on es troben les dades d'entrenament
        File trainData = new File("data/train");
        
        // Divisió de fitxers per accedir a les imatges en el directori d'entrenament
        FileSplit trainSplit = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS);
        
        // Genera etiquetes basant-se en els noms de les carpetes (alfabèticament)
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        
        // Crea un lector de registres per carregar les imatges (redimensionant-les a la mida especificada i assignant etiquetes)
        ImageRecordReader trainRecordReader = new ImageRecordReader(HEIGHT, WIDTH, CHANNELS, labelMaker);
        
        // Defineix una transformació per invertir les imatges horitzontalment
        ImageTransform flipTransform = new FlipImageTransform(1);
        
        // Defineix una transformació per distorsionar les imatges lleugerament
        ImageTransform warpTransform = new WarpImageTransform(0.5f);
        
        // Les transformacions aleatòries permeten augmentar les dades d'entrenament
        // Un vehicle s'ha de detectar com a tal girat horitzontalment o lleugerament distorsionat

        // Crea una transformacions d'imatges amb probabilitats d'aplicació per cada transformació
        List<Pair<ImageTransform, Double>> pipeline = Arrays.asList(
            new Pair<>(flipTransform, 0.5), // 50% de les imatges es fliparan horitzontalment
            new Pair<>(warpTransform, 0.3)  // 30% de les imatges tindran distorsions
        );
        
        // Agrupa les transformacions definides per aplicar-les com a pipeline
        ImageTransform transform = new PipelineImageTransform(pipeline, false);
        
        // Inicialitza el lector de registres amb la divisió de fitxers especificada
        trainRecordReader.initialize(trainSplit);
        
        // Crea un iterador de conjunts de dades per carregar les imatges processades en lots
        DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRecordReader, BATCH_SIZE, 1, NUM_CLASSES);
        
        // Defineix un escalador per normalitzar els valors dels píxels de les imatges a l'interval [0, 1]
        // Normalitza els valors per estabilitzar l'entrenament
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(trainIter);
        trainIter.setPreProcessor(scaler);
        
        // Configura la xarxa neuronal amb diferents capes, incloent convolucions, submostreig i capes denses
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            // Fixa una llavor aleatòria per fer els resultats reproductibles
            .seed(123) 
            // Utilitza l'optimitzador Adam amb una taxa d'aprenentatge de 0.001
            .updater(new Adam(1e-3)) 
            .list()
            // Capa de convolució amb filtres 5x5
            // Extreu característiques com contorns, formes, textures, ...
            .layer(new ConvolutionLayer.Builder(5, 5) 
                .nIn(CHANNELS) // Nombre de canals d'entrada (1 per escala de grisos)
                .stride(1, 1)  // Desplaçament de 1 en ambdues direccions
                .nOut(32)      // 32 filtres de sortida
                .activation(Activation.RELU) // Funció d'activació ReLU
                .build())
            // Capa de submostreig amb pooling màxim
            // Redueix el mapa de característiques essencials per accelerar l'entrenament/classificació
            // Prevé 'overfitting' obligant al model a aprendre característiques essencials
            .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX) 
                .kernelSize(2, 2) // Finestra de pooling de 2x2
                .stride(2, 2)     // Desplaçament de 2 en ambdues direccions
                .build())
            // Segona capa de convolució amb filtres 5x5
            // Ajuda a aprendre característiques més complexes i abstractes de les imatges
            .layer(new ConvolutionLayer.Builder(5, 5)
                .nOut(64)     // 64 filtres de sortida
                .stride(1, 1) // Desplaçament de 1 en ambdues direccions
                .activation(Activation.RELU) // Funció d'activació ReLU
                .build())
            // Segona capa de submostreig amb pooling màxim
            .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX) 
                .kernelSize(2, 2) // Finestra de pooling de 2x2
                .stride(2, 2) // Desplaçament de 2 en ambdues direccions
                .build())
            // Capa densa amb 128 neurones i activació ReLU
            .layer(new DenseLayer.Builder().nOut(128).activation(Activation.RELU).build()) 
            
            // Capa de sortida amb funció de pèrdua log-likelihood
            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(NUM_CLASSES) // Nombre de classes de sortida
                .activation(Activation.SOFTMAX) // Funció d'activació Softmax per obtenir probabilitats de classificació
                .build())
            .setInputType(org.deeplearning4j.nn.conf.inputs.InputType.convolutional(HEIGHT, WIDTH, CHANNELS)) // Defineix el tipus d'entrada per a la xarxa
            .build();
        
        // Crea una xarxa neuronal a partir de la configuració especificada
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        
        // Inicialitza el model
        model.init();
        
        // Assigna un listener per mostrar el "score" cada 10 iteracions durant l'entrenament
        model.setListeners(new ScoreIterationListener(10));
        

        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            System.out.println("Epoch " + (epoch + 1) + " / " + EPOCHS);
            model.fit(trainIter);
        }

        model.save(new File("trained_model.zip"), true);
        System.out.println("Model entrenat i guardat com a 'trained_model.zip'");
    }
}
