import numpy as np
import sigmoidNeuron
import layer
import neuralNetwork

HIDDEN_LAYER1_COUNT =  16
HIDDEN_LAYER2_COUNT = 16
OUTPUT_LAYER_COUNT = 10

# myNeuralNetwork = neuralNetwork.NeuralNetwork([HIDDEN_LAYER1_COUNT, HIDDEN_LAYER2_COUNT, OUTPUT_LAYER_COUNT])

test = neuralNetwork.NeuralNetwork([2, 2])
print(test)

random_pixels = list(np.random.rand(784))
inputs = ([1, 1])

test.forward(inputs) 
print(test)
print("Probabilitys: ", layer.softmax(test.network[1]))