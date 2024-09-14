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

Implementa els següents objectes en JAVA, fent servir el model de clonació prototype i demostra que funciona correctament:

- Una classe Electrodomèstic amb nom, color, preu, marca, eficiència
- Una classe Rentadora que és un electrodomèstic i a més té: revolucions, soroll
- Una classe Nevera que és un electrodomèstic i a més té: frigories, soroll
- Una classe Forn que és un electrodomèstic i a més té: temperatura, autoneteja

A partir de les classes anteriors, crea una llistes de instàncies per cada tipus d’electrodomèstic amb almenys 2 elements de cada tipus (Rentadora, Nevera, Forn), inventa’t els valors

Crea després una llista on cada objecte és un clon dels objectes de la llista principal anterior

Fes dos bucles:

- Compara la llista original amb ella mateixa, compara que són el mateix objecte, de la mateixa classe i tenen iguals dades
- Compara la llista original amb dels clons, compara que són objectes diferents, de la mateixa classe i tenen iguals dades

Les comparacions han de ser de cada element amb l’equivalent de l’altre llista (és a dir 1 a 1, no 1 a tots)
