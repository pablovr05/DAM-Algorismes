package com.neural;

import java.util.function.Function;

public class ActivationFunction {
    private final Function<Double, Integer> activation;
    private final String type;

    public ActivationFunction(Function<Double, Integer> activation, String type) {
        this.activation = activation;
        this.type = type;
    }

    public int activate(double sum) {
        return activation.apply(sum);
    }

    public String getType() {
        return type;
    }

    public static ActivationFunction getActivation(String type) {
        switch (type.toLowerCase()) {
            case "binary_step":
                return new ActivationFunction(sum -> sum >= 0 ? 1 : 0, "binary_step");
            case "sigmoid":
                return new ActivationFunction(sum -> (1 / (1 + Math.exp(-sum)) > 0.5) ? 1 : 0, "sigmoid");
            case "tanh":
                return new ActivationFunction(sum -> Math.tanh(sum) > 0 ? 1 : 0, "tanh");
            case "relu":
                return new ActivationFunction(sum -> (int) (sum > 0 ? sum : 0), "relu");
            case "leaky_relu":
                return new ActivationFunction(sum -> (int) (sum > 0 ? sum : 0.01 * sum), "leaky_relu");
            default:
                throw new IllegalArgumentException("Unknown activation type: " + type);
        }
    }
}
