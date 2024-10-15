package com.project;

import java.util.*;  

public class Main {

    static Scanner in = new Scanner(System.in);

    public static void main(String[] args) {

        String method = "";
        try {
            while (!method.equalsIgnoreCase("ship") 
                && !method.equalsIgnoreCase("truck") 
                && !method.equalsIgnoreCase("van")) {
                System.out.print("Enter a transportation method (ship, truck, van): ");  
                method = in.nextLine(); 
            }

            Transportation obj = FactoryTransportation.getTransportation(method);
            FactoryTransportation.deliver(obj);

        } catch (Error e) {}
    }
}