<div style="display: flex; width: 100%;">
    <div style="flex: 1; padding: 0px;">
        <p>© Albert Palacios Jiménez, 2024</p>
    </div>
    <div style="flex: 1; padding: 0px; text-align: right;">
        <img src="./assets/ieti.png" height="32" alt="Logo de IETI" style="max-height: 32px;">
    </div>
</div>
<br/>

# Multi classificació

Anteriorment, hem vist com les xarxes neuronals es poden utilitzar per resoldre problemes de classificació binària, com identificar si una persona somriu o no. Ara, farem un pas més i explorarem la classificació multi-classe.

La classificació multi-classe consisteix a assignar una entrada a una de múltiples categories possibles. En lloc de dir només "sí" o "no", ara direm la probabilitat que la imatge mostri un objecte.

En aquest tipus de problemes, la capa de sortida de la xarxa neuronal tindrà una neurona per a cada classe, i la xarxa donarà una probabilitat per a cada classe. 

Per exemple, si mostrem imatges de vehicles, la xarxa pot donar aquestes probabilitats:

- Cotxe: 85%
- Moto: 10%
- Vaixell: 5%

La xarxa seleccionarà la classe amb la probabilitat més alta com la seva predicció.


