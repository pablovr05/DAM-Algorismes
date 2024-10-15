package com.project;

public class TransportationShip implements Transportation {

    @Override
    public void deliverPackage() {
        System.out.println("Package is traveling across the ocean");
    }
}
