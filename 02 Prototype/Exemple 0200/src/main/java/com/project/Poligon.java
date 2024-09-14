package com.project;

public abstract class Poligon {
    
    public int x;
    public int y;
    public String color;

    public Poligon() {}

    public Poligon(Poligon target) {
        if (target != null) {
            this.x = target.x;
            this.y = target.y;
            this.color = target.color;
        }
    }

    public abstract Poligon clone();

    @Override
    public boolean equals(Object object2) {
        if (!(object2 instanceof Poligon)) return false;
        Poligon cast2 = (Poligon) object2;
        return cast2.x == x && cast2.y == y && cast2.color.equals(color);
    }
}