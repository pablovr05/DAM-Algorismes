package com.project;

import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;

import org.json.JSONArray;
import org.json.JSONObject;

/*
 * En aquest exemple s'escriu i es llegeix
 * un arxiu Json sense cap model d'accés a dades
 * directament transformant un Objecte a Json
 * i després al revés de Json a Objecte
 */

public class MainJson {

    public static void main(String[] args) {
        String basePath = System.getProperty("user.dir") + "/data/";
        String filePath = basePath + "doraemon.json";

        ArrayList<String> aficions = new ArrayList<>();
        aficions.add("Dormir");
        aficions.add("Jugar");
        aficions.add("Fer-se el llest");

        ObjPersonatge objOut = new ObjPersonatge("Doraemon", "Gat", "cosmic@dorayaki.jp", 1980, aficions);

        try {
            // Escriure un objecte a Json 
            JSONObject jsonObject = new JSONObject();
            jsonObject.put("nom", objOut.getNom());
            jsonObject.put("cognom", objOut.getCognom());
            jsonObject.put("email", objOut.getEmail());
            jsonObject.put("anyNaixement", objOut.getAny());

            JSONArray jsonAficions = new JSONArray();
            for (String aficio : objOut.getAficions()) {
                jsonAficions.put(aficio);
            }
            jsonObject.put("aficions", jsonAficions);

            System.out.println(jsonObject.toString(4)); // 4 es l'espaiat

            FileWriter fileWriter = new FileWriter(filePath);
            fileWriter.write(jsonObject.toString(4));
            fileWriter.flush();
            fileWriter.close();

            // Legir un Json a objecte
            String content = new String(Files.readAllBytes(Paths.get(filePath)));
            JSONObject jsonInput = new JSONObject(content);
            String nom = jsonInput.getString("nom");
            String cognom = jsonInput.getString("cognom");
            String email = jsonInput.getString("email");
            int anyNaixement = jsonInput.getInt("anyNaixement");
            JSONArray jsonAficionsInput = jsonInput.getJSONArray("aficions");

            ArrayList<String> aficionsInput = new ArrayList<>();
            for (int i = 0; i < jsonAficionsInput.length(); i++) {
                aficionsInput.add(jsonAficionsInput.getString(i));
            }

            ObjPersonatge objIn = new ObjPersonatge(nom, cognom, email, anyNaixement, aficionsInput);
            System.out.println("Persona: " + objIn);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}