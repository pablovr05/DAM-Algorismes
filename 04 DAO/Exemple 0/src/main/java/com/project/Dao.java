package com.project;

import java.util.ArrayList;

/*
 * Defineix com es treballarà amb
 * els objectes de dades
 * independenment de com estiguin
 * implemntats a la base de dades
 * 
 * En aquest cas amb les operacions
 * bàsiques CRUD:
 * 
 * Create, Read, Update, Delete
 */

public interface Dao<T> {

    void add(T t); // Equival a Create

    T get(int id); // Equival a Read
    ArrayList<T> getAll();

    void update(int id, T t);

    void delete(int id); 
    
    void print();
}
