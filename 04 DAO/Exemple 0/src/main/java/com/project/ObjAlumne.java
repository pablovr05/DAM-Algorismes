package com.project;

import java.util.ArrayList;

public class ObjAlumne {
    
    private int id;
    private String nom;
    private ArrayList<Integer> cursos;

    public ObjAlumne (int id, String nom, ArrayList<Integer> cursos) {
        this.id = id;
        this.nom = nom;
        this.cursos = cursos;
    }

    public int getId () {
        return id;
    }

    public String getNom () {
        return nom;
    }

    public void setNom (String nom) {
        this.nom = nom;
    }

    public ArrayList<Integer> getCursos() {
        return cursos;
    }

    @Override
    public String toString () {
        return "Alumne: " + this.id + " - " + this.nom + " - " + this.cursos;
    }
}
