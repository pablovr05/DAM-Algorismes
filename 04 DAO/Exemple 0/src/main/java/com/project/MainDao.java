package com.project;

import java.util.ArrayList;

/*
 * En aquest exemple es fa servir
 * l'objecte DAO per gestionar estudiants
 * i cursos amb operacions bàsiques CRUD
 * (Create, Read, Update and Delete)
 * 
 * DAO (Data Access Object) separa la
 * manera que es guarden les dades als arxius
 * de la lògica de treball amb aquestes dades
 */

public class MainDao {
    
    public static String basePath = System.getProperty("user.dir") + "/data/";
    public static String alumnesPath = basePath + "alumnes.json";
    public static String cursosPath = basePath + "cursos.json";

    public static void main(String[] args) {
    
        DaoAlumne daoAlumne = new DaoAlumne();
        DaoCurs daoCurs = new DaoCurs();

        ArrayList<Integer> aCursos = new ArrayList<>();
        aCursos.add(1);
        aCursos.add(2);
        ObjAlumne aIn = new ObjAlumne(100, "Jaimito", aCursos);
        daoAlumne.add(aIn);

        ObjCurs cIn = new ObjCurs(100, "Astrofísica");
        daoCurs.add(cIn);

        System.out.println("Llista d'alumnes:");
        ArrayList<ObjAlumne> llistaAlumnes = daoAlumne.getAll();
        for (int cnt = 0; cnt < llistaAlumnes.size(); cnt = cnt + 1) {
            System.out.println("    " + llistaAlumnes.get(cnt));
        }

        System.out.println("Alumne amb id 2:");
        ObjAlumne alumneGet = daoAlumne.get(2);
        System.out.println("    " + alumneGet);

        System.out.println("Llista de cursos:");
        ArrayList<ObjCurs> llistaCursos = daoCurs.getAll();
        for (int cnt = 0; cnt < llistaCursos.size(); cnt = cnt + 1) {
            System.out.println("    " + llistaCursos.get(cnt));
        }

        System.out.println("Curs amb id 2:");
        ObjCurs cursGet = daoCurs.get(2);
        System.out.println("    " + cursGet);

        // Modificar un alumne (agafar l'objecte, modificar-lo i actualitzar-lo a la bbdd)
        ObjAlumne modAlumne = daoAlumne.get(1);
        modAlumne.setNom(modAlumne.getNom() + "12345");
        daoAlumne.update(modAlumne.getId(), modAlumne);

        daoAlumne.delete(100);
        daoCurs.delete(100);

        System.out.println("Llista d'alumnes: (actualitzada)");        
        daoAlumne.print();

        // Modificar un alumne (agafar l'objecte, modificar-lo i actualitzar-lo a la bbdd)
        modAlumne = daoAlumne.get(1);
        modAlumne.setNom(modAlumne.getNom().replace("12345", ""));
        daoAlumne.update(modAlumne.getId(), modAlumne);
    }
}
