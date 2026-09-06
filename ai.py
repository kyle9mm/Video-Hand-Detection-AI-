import numpy as np
import sigmoidNeuron
import layer
import neuralNetwork

HIDDEN_LAYER1_COUNT =  16
HIDDEN_LAYER2_COUNT = 16
OUTPUT_LAYER_COUNT = 10

myNeuralNetwork = neuralNetwork.NeuralNetwork([HIDDEN_LAYER1_COUNT, HIDDEN_LAYER2_COUNT, OUTPUT_LAYER_COUNT])

print(myNeuralNetwork)