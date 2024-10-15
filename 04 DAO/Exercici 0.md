<div style="display: flex; width: 100%;">
    <div style="flex: 1; padding: 0px;">
        <p>© Albert Palacios Jiménez, 2024</p>
    </div>
    <div style="flex: 1; padding: 0px; text-align: right;">
        <img src="./assets/ieti.png" height="32" alt="Logo de IETI" style="max-height: 32px;">
    </div>
</div>
<br/>

# Exercici 0

Crea un programa “Main.java” que implementi la gestió de dades (en arxius .json) d’un apassionat per la programació. La base de dades estarà en una carpeta tindrà els següents arxius .json:

- **llenguatges.json**, conté una llista amb almenys 5 llenguatges de programació amb atributs:

    id (int, identificador del llenguatge a la base de dades)
    nom (String, nom del llenguatge)
    any (int, any en què va aparèixer)
    dificultat (String dificulttat de programar-lo entre facil, mitja i difícil)
popularitat (int per ordenar-los segons popularitat)

- **eines.json**, conté una llista amb almenys 5 eines de programació amb atributs

    id (int, identificador de la eina la base de dades)
    nom (String, nom de la eina)
    any (int, any en què va aparèixer)
llenguatges (arraylist amb els identificadors dels llenguatges que accepta)

- **software.json**, conté una llista amb almenys 10 programes (o software) coneguts, desenvolupats amb els llenguatges i les eines anteriors

    id (int, identificador del software a la base de dades)
    nom (String, nom del software)
    any (int, any en què va aparèixer)
llenguatges (arraylist amb els identificadors dels llenguatges amb què s’ha desenvolupat)

Un cop tinguis aquesta base de dades:

- Crea els objectes Java que corresponent a cada tipus d’objecte anterior

    (ObjEina.java, ObjLlenguatge.java i ObjSoftware.java)

- Per cada un d’aquests objectes JAVA crea el Dao corresponent basat en CRUD

    (DaoEina.java, DaoLlenguatge.java, DaoSoftware.java)

Fes funcions Update específiques per cada atribut, al seu Dao corresponent, és a dir:

- Dao tindrà add, get, getAll, update, delete, print, setNom, setAny
- Dao tindrà setNom(int id, String nom) i setAny(int id, int any)
- DaoLlenguatge a més tindrà 
    setDificultat(int id, String dificultat)
    setPopularitat(int id, int popularitat)
- DaoEina i DaoSoftware a més tindran:
    setLlenguatgesAdd(int id, int idLlenguatge) Per afegir un id a la llista
    setLlenguatgesDelete(id, int idLlenguatge) per treure un id de la llista

Fes que el programa Main.java tingui aquest codi i mostri aquest resultat:

```java
import java.util.ArrayList;


public class Main {
   public static String basePath = System.getProperty("user.dir") + "/";
   public static String llenguatgesPath = basePath + "./dbProgramacio/llenguatges.json";
   public static String einesPath = basePath + "./dbProgramacio/eines.json";
   public static String softwarePath = basePath + "./dbProgramacio/software.json";


   public static void main(String[] args) {


       DaoEina         daoEina = new DaoEina();
       DaoLlenguatge   daoLlenguatge = new DaoLlenguatge();
       DaoSoftware     daoSoftware = new DaoSoftware();


       // Afegir una eina
       ArrayList<Integer> aLlenguatges0 = new ArrayList<>();
       aLlenguatges0.add(0);
       aLlenguatges0.add(1);
       ObjEina objEina0 = new ObjEina (5, "Text", 2000, aLlenguatges0);
       daoEina.add(objEina0);


       // Modificar una eina
       ArrayList<Integer> aLlenguatges1 = new ArrayList<>();
       aLlenguatges1.add(0);
       aLlenguatges1.add(1);
       aLlenguatges1.add(2);
       ObjEina objEina1 = new ObjEina (5, "TextEdit", 2001, aLlenguatges1);
       daoEina.update(5, objEina1);


       // Afegir llenguatge a eina
       daoEina.setLlenguatgesAdd(5, 3);


       // Treure llenguatge a eina
       daoEina.setLlenguatgesDelete(5, 2);


       // Llistar les eines
       daoEina.print();


       // Esborrar eina amb id 5
       daoEina.delete(5);


       // Afegir un llenguatge
       ObjLlenguatge objLlenguatge0 = new ObjLlenguatge(5, "Dart", 2011, "facil", 8);
       daoLlenguatge.add(objLlenguatge0);


       // Canviar el nom
       daoLlenguatge.setNom(5, "Dart+Flutter");


       // Canviar l'any
       daoLlenguatge.setAny(5, 2018);


       // Canvir dificultat
       daoLlenguatge.setDificultat(5, "mitja");


       // Canviar popularitat
       daoLlenguatge.setPopularitat(5, 9);


       // Llistar els llenguatges
       daoLlenguatge.print();


       // Esborrar llenguatge amb id 5
       daoLlenguatge.delete(5);


       // Afegir un software
       ArrayList<Integer> aLlenguatges2 = new ArrayList<>();
       aLlenguatges2.add(3);
       ObjSoftware objSoftware = new ObjSoftware(10, "webTool", 2022, aLlenguatges2);
       daoSoftware.add(objSoftware);


       // Llistar software
       daoSoftware.print();


       // Esborrar llenguatge amb id 5
       daoSoftware.delete(10);
   }
}
```

<center><img src="./assets/SortidaEx0.png" style="max-width: 90%; width: 400px; max-height: 400px;" alt="">
<br/></center>
<br/>

Sortida esperada:
```text
Eina: 0 Visual Studio Code, 2015 - [0, 1, 2, 3, 4]
Eina: 1 Xcode, 2003 - [0, 1, 2]
Eina: 2 Eclipse, 2001 - [4]
Eina: 3 Vim, 1991 - [0, 1, 2, 3, 4]
Eina: 4 Nano, 1999 - [0, 1, 2, 3, 4]
Eina: 5 TextEdit, 2001 - [0, 1, 3]
Llenguatge: 0 C, 1972 - dificil/8
Llenguatge: 1 C++, 1983 - mitja/7
Llenguatge: 2 Objective C, 1984 - dificil/2
Llenguatge: 3 JavaScript, 1996 - facil/6
Llenguatge: 4 Java, 1995 - mitja/5
Llenguatge: 5 Dart+Flutter, 2018 - mitja/9
Software: 0 Visual Studio Code, 2015 - [0, 1, 2, 3, 4]
Software: 1 Xcode, 2003 - [0, 1, 2]
Software: 2 Eclipse, 2001 - [4]
Software: 3 Vim, 1991 - [0, 1, 2, 3, 4]
Software: 4 Nano, 1999 - [0, 1, 2, 3, 4]
Software: 10 webTool, 2022 - [3]
```