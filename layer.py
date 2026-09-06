import numpy as np
import sigmoidNeuron

class Layer:
    def __init__(self, neuron_count, num_inputs):
        self.neurons = []
        # Create the layer of neurons with random weights and biases 
        for i in range(neuron_count): 
            self.neurons.append(sigmoidNeuron.sigmoidNeuron(num_inputs))


if __name__ == "__main__": # Run this testing code when layer file called directly 
    print("Hello, this is the direct file you are running from")
    input = Layer(5, 2)
    for neuron in input.neurons: 
        print(neuron)
    pass
 