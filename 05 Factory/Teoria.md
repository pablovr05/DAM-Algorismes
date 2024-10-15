<div style="display: flex; width: 100%;">
    <div style="flex: 1; padding: 0px;">
        <p>© Albert Palacios Jiménez, 2024</p>
    </div>
    <div style="flex: 1; padding: 0px; text-align: right;">
        <img src="./assets/ieti.png" height="32" alt="Logo de IETI" style="max-height: 32px;">
    </div>
</div>
<br/>

# Factory

Substituir la creació d’objectes a través de **‘new’** per crides a metodes **‘factory’** específics de cada classe.

Els objectes es segueixen creant amb **‘new’**, però es fa de manera interna al mètode **‘factory’**

El objectes creats a través de **‘factory’** s’anomenen **‘products’**

<center>
<img src="./assets/factory0.png" style="max-width: 90%; width: 400px; max-height: 400px;" alt="">
<br/>
</center>
<br/>

Els clients d’una llibreria no tenen que saber com funciona aquesta llibreria, o fins i tot el tipus d’objecte o constructors que es fan servir.

Amb factory tens un **‘product’** que declara una interfície comú a tots els objectes i les seves subclasses

Cal que hi hagi una classe **‘creator’** que té el mètode per fabricar **‘products’**

<center>
<img src="./assets/factory1.png" style="max-width: 90%; width: 400px; max-height: 400px;" alt="">
<br/>
</center>
<br/>

## Quan es fa servir Factory

- Quan no sabem amb quins tipus d’objectes i dependencies haurà de treballar el nostre codi

- Quan volem donar maneres d’extendre els components de la nostre llibreria

- Quan volem estalviar recursos del sistema, aprofitant objectes enlloc de refentlos cada vegada

En Java Factory es pot implementar a través de:

- **interface** i classes que implementen les seves funcions

- Amb **classes derivades** (extends) i sobreescrivint les funcions necessàries

## Exemple

En aquest exemple veiem que es crea un tipus d'objecte o un altre segons la consulta.

Per fer-ho es crida a **getTransportation** enlloc de al constructor directament.

```java
public class FactoryTransportation {

    public static Transportation getTransportation(String method) {

		if("ship".equalsIgnoreCase(method)) 
            return new TransportationShip();
		else if("truck".equalsIgnoreCase(method)) 
            return new TransportationTruck();
        else if("van".equalsIgnoreCase(method)) 
            return new TransportationVan();

        return null;
	}

    public static void deliver (Transportation transport) {
        transport.deliverPackage();
    }
}
```