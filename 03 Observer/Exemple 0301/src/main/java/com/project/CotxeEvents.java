package com.project;

import java.beans.PropertyChangeListener;
import java.beans.PropertyChangeSupport;

public class CotxeEvents {
    private PropertyChangeSupport llistaObservers = new PropertyChangeSupport(this);

    private CotxeEstats estat;
    private int gasolina;

    CotxeEvents () {
        this.estat = CotxeEstats.ATURAT;
        this.gasolina = 3;
    }

    public void addPropertyChangeListener(String name, PropertyChangeListener listener) {
        llistaObservers.addPropertyChangeListener(name, listener);
    }

    public void removePropertyChangeListener(String name, PropertyChangeListener listener) {
        llistaObservers.removePropertyChangeListener(name, listener);
    }

    public void setEstat (CotxeEstats newValue) {
        CotxeEstats oldValue = this.estat;
        if (oldValue != newValue) {
            if (newValue != CotxeEstats.ATURAT) {
                this.setGasolina(this.gasolina - 1);
            }
            if (this.gasolina > 0) {
                this.estat = newValue;
                llistaObservers.firePropertyChange("estat", oldValue, newValue);
            }
        }
    }

    public void setGasolina (int newValue) {
        int oldValue = this.gasolina;
        if (newValue > 0) {
            this.gasolina = newValue;
        } else {
            this.gasolina = 0;
            this.setEstat(CotxeEstats.ATURAT);
        }
        if (oldValue != this.gasolina) {
            llistaObservers.firePropertyChange("gasolina", oldValue, newValue);
        }
    }
}