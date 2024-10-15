package com.project;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;

public class DaoCurs implements Dao<ObjCurs> {

    private void writeList(ArrayList<ObjCurs> llista) {
        try {
            JSONArray jsonArray = new JSONArray();
            for (ObjCurs curs : llista) {
                JSONObject jsonObject = new JSONObject();
                jsonObject.put("id", curs.getId());
                jsonObject.put("nom", curs.toString());
                jsonArray.put(jsonObject);
            }

            PrintWriter out = new PrintWriter(MainDao.cursosPath);
            out.write(jsonArray.toString(4));  // 4 es l'espaiat
            out.flush();
            out.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    private int getPosition(int id) {
        int result = -1;
        ArrayList<ObjCurs> llista = getAll();
        for (int cnt = 0; cnt < llista.size(); cnt = cnt + 1) {
            ObjCurs curs = llista.get(cnt);
            if (curs.getId() == id) {
                result = cnt;
                break;
            }
        }
        return result;
    }

    @Override
    public void add(ObjCurs curs) {
        ArrayList<ObjCurs> llista = getAll();
        ObjCurs item = get(curs.getId());
        if (item == null) {
            llista.add(curs);
            writeList(llista);
        }
    }

    @Override
    public ObjCurs get(int id) {
        ObjCurs result = null;
        ArrayList<ObjCurs> llista = getAll();
        int pos = getPosition(id);
        if (pos != -1) {
            result = llista.get(pos);
        }
        return result;
    }

    @Override
    public ArrayList<ObjCurs> getAll() {
        ArrayList<ObjCurs> result = new ArrayList<>();
        try {
            String content = new String(Files.readAllBytes(Paths.get(MainDao.cursosPath)));
            JSONArray jsonArray = new JSONArray(content);
            for (int i = 0; i < jsonArray.length(); i++) {
                JSONObject jsonObject = jsonArray.getJSONObject(i);
                int id = jsonObject.getInt("id");
                String nom = jsonObject.getString("nom");
                result.add(new ObjCurs(id, nom));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return result;
    }

    @Override
    public void update(int id, ObjCurs curs) {
        ArrayList<ObjCurs> llista = getAll();
        int pos = getPosition(id);
        if (pos != -1) {
            llista.set(pos, curs);
            writeList(llista);
        }
    }

    @Override
    public void delete(int id) {
        ArrayList<ObjCurs> llista = getAll();
        int pos = getPosition(id);
        if (pos != -1) {
            llista.remove(pos);
            writeList(llista);
        }
    }

    @Override
    public void print() {
        ArrayList<ObjCurs> llista = getAll();
        for (int cnt = 0; cnt < llista.size(); cnt = cnt + 1) {
            System.out.println("    " + llista.get(cnt));
        }
    }
}
