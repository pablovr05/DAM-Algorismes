#!/usr/bin/env python3

from perceptron import initialize_weights, train, test_accuracy

# Provar amb diferents EPOCHS: 0, 1, ... 10 per veure el % d'encert

EPOCHS = 10 
LEARNING_RATE = 0.1
INPUT_SIZE = 4

inputs = [
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 0],
    [0, 1, 0, 1],
    [0, 1, 1, 0],
    [0, 1, 1, 1]
]

labels = [0, 1, 0, 1, 0, 1, 0, 1]

weights = initialize_weights(INPUT_SIZE)
bias = 0.0

weights, bias = train(weights, bias, inputs, labels, LEARNING_RATE, EPOCHS)
accuracy = test_accuracy(weights, bias, inputs, labels)
print(f"\nPercentatge d'encert del Perceptró entrenat amb {EPOCHS} EPOCHS: {accuracy}%\n")
