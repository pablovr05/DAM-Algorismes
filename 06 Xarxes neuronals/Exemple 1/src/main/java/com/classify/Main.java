package com.classify;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;

public class Main {

    private static final int WIDTH = 64;
    private static final int HEIGHT = 64;
    private static final int CHANNELS = 3;

    public static void classifyAndCheck(MultiLayerNetwork model, String imagePath, boolean expectedSmile) throws IOException {

        // Selecciona la imatge per fer la predicció
        File testFile = new File(imagePath);
    
        if (testFile.exists()) {
            // Càrrega i normalitza la imatge
            NativeImageLoader loader = new NativeImageLoader(HEIGHT, WIDTH, CHANNELS);
            INDArray image = loader.asMatrix(testFile);
            ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
            scaler.transform(image);
    
            // Predicció
            INDArray output = model.output(image); 
            // output és un array on cada etiqueta té la probabilitat d'encert 
            // output[0] = probabilitat que sigui 'non_smile'
            // output[1] = probabilitat que sigui 'smile'
            // L'ordre és 'non_smile', 'smile' perquè s'ordena el nom de les carpetes alfabèticament

            INDArray predictionArray = Nd4j.argMax(output, 1) 
            // predictionArray array d'un sol element amb l'index de la probabilitat més alta de 'output'

            int predictedClass = predictionArray.getInt(0);
            // index de l'etiqueta amb més probabilitat (0 per 'non_smile', 1 per 'smile')

            boolean predictedSmile = (predictedClass == 1);
    
            boolean isCorrect = (predictedSmile == expectedSmile);           
            System.out.println("Resultat \"" + testFile.getName() + "\": " + (predictedSmile ? "smile" : "non_smile") + " > " + (isCorrect ? "Correcte" : "Incorrecte"));
        } else {
            System.out.println("L'arxiu de test no es troba: " + testFile.getPath());
        }
    }
   
    public static void main(String[] args) throws IOException {

        // Carregar el model entrenat
        MultiLayerNetwork model = MultiLayerNetwork.load(new File("trained_model.zip"), true);
        System.out.println("Model carregat correctament.");

        classifyAndCheck(model, "data/test/George_W_Bush_0047.jpg", false);
        classifyAndCheck(model, "data/test/Abdullah_0001.jpg", false);
        classifyAndCheck(model, "data/test/Bill_Paxton_0002.jpg", true);
        classifyAndCheck(model, "data/test/Annette_Bening_0002.jpg", true);
        classifyAndCheck(model, "data/test/Elizabeth_Regan_0001.jpg", true);
        classifyAndCheck(model, "data/test/Nancy_Pelosi_0001.jpg", false);
    }
}
