import random

def initialize_weights(input_size):
    return [random.uniform(-0.5, 0.5) for _ in range(input_size)]

def relu(x):
    return 1 if x > 0 else 0

def predict(weights, bias, inputs):
    total = bias + sum(w * i for w, i in zip(weights, inputs))
    return relu(total)

def train(weights, bias, input_data, labels, learning_rate, epochs):
    for _ in range(epochs):
        for inputs, label in zip(input_data, labels):
            prediction = predict(weights, bias, inputs)
            error = label - prediction
            for j in range(len(weights)):
                weights[j] += learning_rate * error * inputs[j]
            bias += learning_rate * error
    return weights, bias

def test_accuracy(weights, bias, input_data, labels):
    correct = 0
    for inputs, label in zip(input_data, labels):
        prediction = predict(weights, bias, inputs)
        correct += (prediction == label)
        print(f"Input: {''.join(map(str, inputs))} -> Result: {prediction} ({'ok' if prediction == label else 'ko'})")
    return (correct / len(input_data)) * 100
