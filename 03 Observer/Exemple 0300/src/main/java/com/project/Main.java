package com.project;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.concurrent.TimeUnit;

/*
 * Aquest exemple defineix un
 * model observer simple, sobre
 * tipus de dades primitius
 * 
 * També mira els canvis a l'arxiu
 * 'arxiu.txt' i els mostra
 * per pantalla quan es modifica
 * encara que siguin fets per una
 * eina externa
 */

public class Main {

    public static void main (String[] args) {
        try {
            File baseDir = new File("./data/");

            if (!baseDir.exists()) {
                baseDir.mkdirs(); 
            }

            // Ruta de l'arxiu
            String filePath = baseDir.getPath() + "/arxiu.txt";
            File arxiu = new File(filePath);

            // Crear l'arxiu si no existeix
            if (!arxiu.exists()) {
                arxiu.createNewFile();
            }

            Observable<Integer> obsNum = new Observable<Integer>(0) {
                @Override
                public void propertyChange(Integer oldValue, Integer newValue) {
                    System.out.printf("obsNum ha canviat de %s cap a %s\n", oldValue, newValue);
                }            
            };

            Observable<String> obsTxt = new Observable<String>("poma") {
                @Override
                public void propertyChange(String oldValue, String newValue) {
                    System.out.printf("obsTxt ha canviat de %s cap a %s\n", oldValue, newValue);
                }            
            };

            ObservableFile obsFile = new ObservableFile(arxiu) {
                @Override
                public void onChange() {
                    System.out.println("Arxiu modificat");
                }
            };

            obsNum.setValue(1);
            obsTxt.setValue("llimona");
            obsNum.setValue(2);
            obsTxt.setValue("meló");

            System.out.println("Esperem 10 segons per si hi ha canvis a l'arxiu 'arxiu.txt'");
            for (int cnt = 10; cnt > 0; cnt = cnt - 1) {
                System.out.println("Contador: " + cnt);
                wait(1);
                if (cnt == 9) {
                    escriuArxiu(arxiu, "cnt 9");
                } else if (cnt == 6) {
                    // Escribim l'arxiu per comprovar que detecta l'event
                    escriuArxiu(arxiu, "cnt 6");
                } else if (cnt == 1) {
                    System.out.printf("Espera uns segons ...");
                }
            }
        } catch (IOException e) {
            e.printStackTrace();  // Captura l'excepció IOException
        }
    }

    public static void wait (int seconds) {
        try {
            TimeUnit.SECONDS.sleep(seconds);
        } catch (InterruptedException e) { e.printStackTrace(); }
    }

    public static void escriuArxiu (File arxiu, String valor) {
        try {
            FileWriter fileWriter = new FileWriter(arxiu);
            fileWriter.write(valor);
            fileWriter.close();
        } catch (IOException e) { e.printStackTrace(); }
    }
}
