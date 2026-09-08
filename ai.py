import numpy as np
import sigmoidNeuron
import layer
import neuralNetwork
from sklearn.datasets import fetch_openml

HIDDEN_LAYER1_COUNT =  16
HIDDEN_LAYER2_COUNT = 16
OUTPUT_LAYER_COUNT = 10

mnist = fetch_openml('mnist_784', version=1)
first_image = np.array(mnist.data.iloc[0]) # Get row of first image 
first_image = first_image/255 # Normalise first image 

test = neuralNetwork.NeuralNetwork([10, 10])
print(test)

test.forward(first_image) 
print(test)
print("Probabilitys: ", layer.softmax(test.network[1]))