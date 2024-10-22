package com.classify;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;

public class Main {

    private static final int WIDTH = 128;
    private static final int HEIGHT = 128;
    private static final int CHANNELS = 1;
    private static final String[] VEHICLE_CLASSES = { 
        // En el mateix ordre que s'ha generat l'entrenament (alfabètic)
        "bus", "family_sedan", "fire_engine", "heavy_truck", 
        "jeep", "minibus", "racing_car", "SUV", "taxi", "truck"
    };
    
    public static void classifyAndCheck(MultiLayerNetwork model, String imagePath, String label) throws IOException {

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
            // output[0] = probabilitat que sigui 'bus'
            // output[1] = probabilitat que sigui 'family_sedan'
            // ...

            INDArray predictionArray = Nd4j.argMax(output, 1);
            // predictionArray array d'un sol element amb l'index de la probabilitat més alta de 'output'

            int predictedClass = predictionArray.getInt(0);
            // index de l'etiqueta amb més probabilitat (0 per 'bus', 1 per 'family_sedan', ...)
            
            // Mostra la probabilitat de ser cada tipus de vehicle
            System.out.println("Resultat '" + testFile.getName() + "'':");
            for (int i = 0; i < VEHICLE_CLASSES.length; i++) {
                System.out.printf("  - %s: %.2f%%\n", VEHICLE_CLASSES[i], output.getFloat(i) * 100);
            }
            System.out.println("  Predicció: '" + VEHICLE_CLASSES[predictedClass] + "'' - Esperat: '" + label + "'\n");
        } else {
            System.out.println("L'arxiu de test no es troba: " + testFile.getPath());
        }
    }
   
    public static void main(String[] args) throws IOException {

        // Carregar el model entrenat
        MultiLayerNetwork model = MultiLayerNetwork.load(new File("trained_model.zip"), true);
        System.out.println("Model carregat correctament.");

        // Exemple de validació d'arxius a la carpeta de test
        classifyAndCheck(model, "data/test/fe6ba3df4ec47d2107d4aed4844d0ca9.jpg", "racing_car");
        classifyAndCheck(model, "data/test/f550965cf9cc6f07184923a32d728e94.jpg", "taxi");
        classifyAndCheck(model, "data/test/d9158bb9ab1810deaf519d826959bb4c.jpg", "fire_engine");
        classifyAndCheck(model, "data/test/7678977975a2ab3fc51efbafd0baca37.jpg", "jeep");
        classifyAndCheck(model, "data/test/21fe41471f49f75b7a8fec97e7914e07.jpg", "truck");
    }
}
