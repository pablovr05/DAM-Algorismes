package com.project;

public class TransportationTruck implements Transportation {

    @Override
    public void deliverPackage() {
        System.out.println("Highway traveling");
    }
}
