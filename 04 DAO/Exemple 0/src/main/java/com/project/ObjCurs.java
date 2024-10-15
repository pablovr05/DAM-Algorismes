package com.project;

public class ObjCurs {

    private int id;
    private String nom;

    public ObjCurs (int id, String nom) {
        this.id = id;
        this.nom = nom;
    }

    public int getId () {
        return id;
    }

    @Override
    public String toString () {
        return "Curs: " + this.id + " - " + this.nom;
    }
}
