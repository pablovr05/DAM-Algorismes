package com.project;

import java.util.ArrayList;

public class ObjPersonatge {
    private String nom;
    private String cognom;
    private String email;
    private int any;
    private ArrayList<String> aficions;

    public ObjPersonatge(String nom, String cognom, String email, int any, ArrayList<String> aficions) {
        this.nom = nom;
        this.cognom = cognom;
        this.email = email;
        this.any = any;
        this.aficions = aficions;
    }

    public String getNom() {
        return nom;
    }

    public void setNom(String nom) {
        this.nom = nom;
    }

    public String getCognom() {
        return cognom;
    }

    public void setCognom(String cognom) {
        this.cognom = cognom;
    }

    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email;
    }

    public int getAny() {
        return any;
    }

    public void setAny(int any) {
        this.any = any;
    }

    public ArrayList<String> getAficions() {
        return aficions;
    }

    public void setAficions(ArrayList<String> aficions) {
        this.aficions = aficions;
    }

    @Override
    public String toString() {
        return this.nom + " " + this.cognom + ", " + this.any + "(" + this.email + ") " + this.aficions;
    }
}