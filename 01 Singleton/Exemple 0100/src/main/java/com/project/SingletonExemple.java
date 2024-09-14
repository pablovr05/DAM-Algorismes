package com.project;

public final class SingletonExemple {
    
    private static SingletonExemple instance;
    public String value;

    private SingletonExemple(String value) {
        // Simulem una inicialització lenta
        try {
            Thread.sleep(1000);
        } catch (InterruptedException ex) {
            ex.printStackTrace();
        }
        this.value = value;
    }

    public static SingletonExemple getInstance(String value) {
        if (instance == null) {
            instance = new SingletonExemple(value);
        }
        return instance;
    }
}