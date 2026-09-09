import numpy as np
import random 


class SigmoidNeuron: # Class for creating an instance of sigmoid neuron 
    def __init__(self, num_inputs): 
        self.weights = np.random.rand(num_inputs) - 0.5 # Create an array of randoms numbers between 0 - 1 for each weight of length number of inputs
        self.bias = random.random() # Create a random number for bias between 0 and 1

    def __str__(self):
        return f"Neuron: Weights={self.weights}, Bias={self.bias}"

