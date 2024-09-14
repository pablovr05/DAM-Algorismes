package com.project;

import java.util.ArrayList;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        
        List<Poligon> llista = new ArrayList<>();
        List<Poligon> llistaCopy = new ArrayList<>();

        Cercle circle = new Cercle();
        circle.x = 10;
        circle.y = 20;
        circle.radius = 15;
        circle.color = "red";
        llista.add(circle);

        Cercle anotherCircle = (Cercle) circle.clone();
        llista.add(anotherCircle);

        Rectangle rectangle = new Rectangle();
        rectangle.width = 10;
        rectangle.height = 20;
        rectangle.color = "blue";
        llista.add(rectangle);

        // Clonar la llista
        for (Poligon obj : llista) {
            llistaCopy.add(obj.clone());
        }

        System.out.println("Comparar la mateixa llista:");
        for (int i = 0; i < llista.size(); i++) {
            compare(i, llista.get(i), llista.get(i));
        }

        System.out.println("Comparar amb la llista clonada:");
        for (int i = 0; i < llista.size(); i++) {
            compare(i, llista.get(i), llistaCopy.get(i));
        }

        System.out.println("Comparar amb la llista clonada però invertida:");
        for (int i = 0; i < llista.size(); i++) {
            compare(i, llista.get(i), llistaCopy.get(llista.size() - i - 1));
        }
    }

    static void compare (int i, Poligon a, Poligon b) {
        if (a == b) {
            System.out.println(i + ": Els poligons són el mateix objecte");
        } else {
            System.out.print(i + ": Els poligons són objectes diferents - ");
            if (a.equals(b)) {
                System.out.println(i + ": Els poligons són idèntics");
            } else {
                System.out.println(i + ": Els poligons NO són identics");
            }
        }
    }
}