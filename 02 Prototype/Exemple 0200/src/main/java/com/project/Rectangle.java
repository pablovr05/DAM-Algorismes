package com.project;

public class Rectangle extends Poligon {
    public int width;
    public int height;

    public Rectangle() {
    }

    public Rectangle(Rectangle target) {
        super(target);
        if (target != null) {
            this.width = target.width;
            this.height = target.height;
        }
    }

    @Override
    public Poligon clone() {
        return new Rectangle(this);
    }

    @Override
    public boolean equals(Object object2) {
          // Comprova si són la mateixa referència
        if (this == object2) return true;
        
        if (!(object2 instanceof Poligon) || !super.equals(object2)) return false;
    
        // Comprova si els dos objectes són exactament de la mateixa classe
        if (!this.getClass().equals(object2.getClass())) return false;
    
        Rectangle cast2 = (Rectangle) object2;
        return cast2.width == width && cast2.height == height;
    }
}