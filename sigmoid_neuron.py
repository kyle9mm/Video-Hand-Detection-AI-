import numpy as np
import random 
 

class sigmoidNeuron: # Class for creating an instance of sigmoid neuron 
    def __init__(self, num_inputs): 
        self.weights = np.random.rand(1, num_inputs) # Create an array of randoms numbers between 0 - 1 for each weight of length number of inputs
        self.bias = random.random() # Create a random number for bias between 0 and 1

    def output(): 
        print("Hello, getting the output for you")

 
if __name__ == "__main__": # Run this testing code when sigmoid neuron file called directly 
    print("Hello, this is the direct file you are running from")
    neuron = sigmoidNeuron(5) 
    print("The random weights are ", neuron.weights)
    print("The Bias is ", neuron.bias)
    pass
 