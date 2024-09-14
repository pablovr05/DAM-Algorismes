package com.project;

import java.lang.reflect.Constructor;

public class Main {

    public static void main(String[] args) {

        System.out.println("Aquest exemple:");
        System.out.println("- Triga una estona a iniciar l'objecte");
        System.out.println("- Mostra que 1 i 2 tenen la mateixa instància de l'objecte (Hola, Adeu)");
        System.out.println("- Mostra que 3 no té la mateixa instància (Pepito)");
        System.out.println("");

        System.out.println(".. .iniciant 1 ...");
        SingletonExemple instance1 = SingletonExemple.getInstance("Hola");
        System.out.println(instance1.value);

        System.out.println(".. .iniciant 2 ...");
        SingletonExemple instance2 = SingletonExemple.getInstance("Adeu");
        System.out.println(instance2.value);

        System.out.println(".. .iniciant 3 ...");
        SingletonExemple instance3 = getNewDestroyedInstance("Pepito");
        System.out.println(instance3.value);
    }

    static SingletonExemple getNewDestroyedInstance (String value) {
        
        SingletonExemple result = null;
        try {
            Constructor<?>[] constructors = SingletonExemple.class.getDeclaredConstructors();
            for (Constructor<?> constructor : constructors) {
                //Below code will destroy the singleton pattern
                constructor.setAccessible(true);
                result = (SingletonExemple) constructor.newInstance(value);
                break;
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return result;
    }
}