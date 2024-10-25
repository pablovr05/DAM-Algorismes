package com.project;

import java.util.*;

interface ActivationFunction {
    int activate(double sum);

    static ActivationFunction getActivation(String type) {
        switch (type.toLowerCase()) {
            case "binary_step":
                return sum -> sum >= 0 ? 1 : 0;
            case "sigmoid":
                return sum -> (int) (1 / (1 + Math.exp(-sum)) > 0.5 ? 1 : 0);
            case "tanh":
                return sum -> (int) (Math.tanh(sum) > 0 ? 1 : 0);
            case "relu":
                return sum -> sum > 0 ? (int) sum : 0;
            case "leaky_relu":
                return sum -> sum > 0 ? (int) sum : (int) (0.01 * sum);
            default:
                throw new IllegalArgumentException("Unknown activation type: " + type);
        }
    }
}