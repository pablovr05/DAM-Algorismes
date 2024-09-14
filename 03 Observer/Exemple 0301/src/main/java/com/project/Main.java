package com.project;

import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;

/*
 * En aquest exmple fa servir el model Observer
 * es veu com afegir funcions que
 * es criden segons canvis de variables
 * en un objecte
 * Es poden definir diverses funcions (exemple gasolina)
 * a un mateix canvi de variable
 */

public class Main {

    public static void main (String[] args) {

        CotxeEvents cotxe = new CotxeEvents();

        PropertyChangeListener l0 = new PropertyChangeListener() {
            @Override
            public void propertyChange(PropertyChangeEvent evt) {
                System.out.printf("L'estat s'ha canviat de %s cap a %s a través de la propietat '%s'\n",
                                  evt.getOldValue(), evt.getNewValue(), evt.getPropertyName());                
            }
        };
        cotxe.addPropertyChangeListener("estat", l0);

        PropertyChangeListener l1 = new PropertyChangeListener() {
            @Override
            public void propertyChange(PropertyChangeEvent evt) {
                System.out.printf("La gasolina s'ha canviat de %s cap a %s a través de la propietat '%s'\n",
                                  evt.getOldValue(), evt.getNewValue(), evt.getPropertyName());                
            }
        };
        cotxe.addPropertyChangeListener("gasolina", l1);

        PropertyChangeListener l2 = new PropertyChangeListener() {
            @Override
            public void propertyChange(PropertyChangeEvent evt) {
                System.out.printf("Gasolina segon event\n");                
            }
        };
        cotxe.addPropertyChangeListener("gasolina", l2);

        cotxe.setEstat(CotxeEstats.ENDAVANT);
        cotxe.setEstat(CotxeEstats.EDARRERA);

        // Les següents accions ja no les pot fer perquè no té gasolina
        cotxe.setEstat(CotxeEstats.GIRDRETA);
        cotxe.setEstat(CotxeEstats.EDARRERA);
        cotxe.setEstat(CotxeEstats.ATURAT);
    }
}