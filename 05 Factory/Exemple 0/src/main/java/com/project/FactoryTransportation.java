package com.project;

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
