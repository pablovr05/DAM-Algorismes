package com.project;

import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;

import org.json.JSONArray;
import org.json.JSONObject;

/*
 * Implementa la relació entre
 * el model DAO basat en CRUD
 * i el model de la base de dades real
 */

public class DaoAlumne implements Dao<ObjAlumne> {

    private void writeList(ArrayList<ObjAlumne> llista) {
        try {
            JSONArray jsonArray = new JSONArray();
            for (ObjAlumne alumne : llista) {
                JSONObject jsonObject = new JSONObject();
                jsonObject.put("id", alumne.getId());
                jsonObject.put("nom", alumne.getNom());
                JSONArray jsonCursos = new JSONArray(alumne.getCursos());
                jsonObject.put("cursos", jsonCursos);
                jsonArray.put(jsonObject);
            }
            PrintWriter out = new PrintWriter(MainDao.alumnesPath);
            out.write(jsonArray.toString(4)); // 4 es l'espaiat
            out.flush();
            out.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private int getPosition (int id) {
        int result = -1;
        ArrayList<ObjAlumne> llista = getAll();
        for (int cnt = 0; cnt < llista.size(); cnt = cnt + 1) {
            ObjAlumne alumne = llista.get(cnt);
            if (alumne.getId() == id) {
                result = cnt;
                break;
            }
        }
        return result;
    }

    @Override
    public void add(ObjAlumne alumne) {
        ArrayList<ObjAlumne> llista = getAll();
        ObjAlumne item = get(alumne.getId());
        if (item == null) {
            llista.add(alumne);
            writeList(llista);
        }
    }

    @Override
    public ObjAlumne get(int id) {
        ObjAlumne result = null;
        ArrayList<ObjAlumne> llista = getAll();
        int pos = getPosition(id);
        if (pos != -1) {
            result = llista.get(pos);
        }
        return result;
    }

    @Override
    public ArrayList<ObjAlumne> getAll() {
        ArrayList<ObjAlumne> result = new ArrayList<>();
        try {
            String content = new String(Files.readAllBytes(Paths.get(MainDao.alumnesPath)));
            
            JSONArray jsonArray = new JSONArray(content);
            for (int i = 0; i < jsonArray.length(); i++) {
                JSONObject jsonObject = jsonArray.getJSONObject(i);
                int id = jsonObject.getInt("id");
                String nom = jsonObject.getString("nom");
                JSONArray jsonCursos = jsonObject.getJSONArray("cursos");
                ArrayList<Integer> cursos = new ArrayList<>();
                for (int j = 0; j < jsonCursos.length(); j++) {
                    cursos.add(jsonCursos.getInt(j));
                }
                ObjAlumne alumne = new ObjAlumne(id, nom, cursos);
                result.add(alumne);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return result;
    }
    

    @Override
    public void update(int id, ObjAlumne alumne) {
        ArrayList<ObjAlumne> llista = getAll();
        int pos = getPosition(id);
        if (pos != -1) {
            llista.set(pos, alumne);
            writeList(llista);
        }
    }

    @Override
    public void delete(int id) {
        ArrayList<ObjAlumne> llista = getAll();
        int pos = getPosition(id);
        if (pos != -1) {
            llista.remove(pos);
            writeList(llista);
        }
    }

    @Override
    public void print () {
        ArrayList<ObjAlumne> llista = getAll();
        for (int cnt = 0; cnt < llista.size(); cnt = cnt + 1) {
            System.out.println("    " + llista.get(cnt));
        }
    }
}
