<div style="display: flex; width: 100%;">
    <div style="flex: 1; padding: 0px;">
        <p>© Albert Palacios Jiménez, 2024</p>
    </div>
    <div style="flex: 1; padding: 0px; text-align: right;">
        <img src="./assets/ieti.png" height="32" alt="Logo de IETI" style="max-height: 32px;">
    </div>
</div>
<br/>

# Singleton

El patró de disseny **Singleton** assegura que només hi ha una instància d’una classe determinada

Crees un objecte, si al cap d’una estona tones a crear un objecte d’aquella classe reps l'original enlloc d'una nova instància

Aquest comportament, permet accedir un objecte de manera global en un programa, protegint-lo de sobre-escriptures

## Crear un objecte Singleton

Posar el constructor de la classe com a **‘private’**

Fer un mètode de creació ‘static’ que funciona com a constructor, però:

- El primer cop crida al constructor privat

- La resta de vegades, retorna l’objecte creat originalment

```java
public final class ExempleSingleton {
  
    private static ExempleSingleton instance;
    public String value;

    // Constructor 'privat'
    private ExempleSingleton(String value) {
        this.value = value;
    }

    public static ExempleSingleton getInstance(String value) {
        if (instance == null) {
            instance = new ExempleSingleton(value);
       }
       return instance;
   }
}
```

### Avantatges

- Assegura que només hi ha una instància de la classe
- Es pot accedir a l’objecte de manera global
- Només s’inicia el primer cop

### Inconvenients

- Es pot considerar un ‘anti-patró’ si amaga un mal disseny de l’aplicació
- En entorns amb threads cal assegurar que diversos fils no creen un objecte ‘singelton’ diverses vegades

## Ignorar Singletons

A vegades ens interessa obtenir diverses instàncies d'un objecte Singleton

El següent codi aconsegueix destruir el patró de Singleton, obtenint dues instàncies

```java
ExempleSingleton instanceOne = ExempleSingleton.getInstance("Hola");
ExempleSingleton instanceTwo = null;
try {
    Constructor[] constructors = ExempleSingleton.class.getDeclaredConstructors();
    for (Constructor constructor : constructors) {
        //Below code will destroy the singleton pattern
        constructor.setAccessible(true);
        instanceTwo = (ExempleSingleton) constructor.newInstance("Adeu");
        break;
    }
} catch (Exception e) { e.printStackTrace(); }
```

### Objectes Singleton i Serializable

Tot i que hi ha maneres d’implementar Singleton en objectes serialitzables, és fàcil trencar l’esquema i crear diverses instàncies serialitzant i des-serialitzant els objectes cada vegada que es vol una instància nova.