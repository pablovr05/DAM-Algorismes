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

    private static final int EPOCHS = 25;
    private static final int WIDTH = 128;
    private static final int HEIGHT = 128;
    private static final int CHANNELS = 1;
    private static final int BATCH_SIZE = 32;
    private static final int NUM_CLASSES = 10;

    public static void main(String[] args) throws Exception {

        File trainData = new File("data/train");
        FileSplit trainSplit = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS);
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        ImageRecordReader trainRecordReader = new ImageRecordReader(HEIGHT, WIDTH, CHANNELS, labelMaker);

        ImageTransform flipTransform = new FlipImageTransform(1);
        ImageTransform warpTransform = new WarpImageTransform(0.5f);
        List<Pair<ImageTransform, Double>> pipeline = Arrays.asList(
            new Pair<>(flipTransform, 0.5), // 50% de les imatges es fliparan horitzontalment
            new Pair<>(warpTransform, 0.3)  // 30% de les imatges tindran distorsions
        );
        ImageTransform transform = new PipelineImageTransform(pipeline, false);


        trainRecordReader.initialize(trainSplit);
        DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRecordReader, BATCH_SIZE, 1, NUM_CLASSES);

        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(trainIter);
        trainIter.setPreProcessor(scaler);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(123)
            .updater(new Adam(1e-3))
            .list()
            .layer(new ConvolutionLayer.Builder(5, 5)
                .nIn(CHANNELS)
                .stride(1, 1)
                .nOut(32)
                .activation(Activation.RELU)
                .build())
            .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build())
            .layer(new ConvolutionLayer.Builder(5, 5)
                .nOut(64)
                .stride(1, 1)
                .activation(Activation.RELU)
                .build())
            .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build())
            .layer(new DenseLayer.Builder().nOut(128).activation(Activation.RELU).build())
            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(NUM_CLASSES)
                .activation(Activation.SOFTMAX)
                .build())
            .setInputType(org.deeplearning4j.nn.conf.inputs.InputType.convolutional(HEIGHT, WIDTH, CHANNELS))
            .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));

        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            System.out.println("Epoch " + (epoch + 1) + " / " + EPOCHS);
            model.fit(trainIter);
        }

        model.save(new File("trained_model.zip"), true);
        System.out.println("Model entrenat i guardat com a 'trained_model.zip'");
    }
}
