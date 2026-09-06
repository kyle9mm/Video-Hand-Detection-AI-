import numpy as np
import sigmoidNeuron

class Layer:
    def __init__(self, neuron_count, num_inputs):
        self.neurons = []
        # Create the layer of neurons with random weights and biases 
        for i in range(neuron_count): 
            self.neurons.append(sigmoidNeuron.sigmoidNeuron(num_inputs))

  
def softmax(layer): # Use softmax to calculate the probability of each neuron as being the most likely correct answer
    zeta = []
    probabilitys = []
    # Go through all neurons in the layer and store their z values in zeta
    for neuron in layer.neurons:
        zeta.append(neuron.z)
    # Calculate sum of all exponentials to the power of zeta 
    exp_values = np.exp(zeta) 
    total = sum(exp_values)
    # Go through each zeta value for each neuron and calculate its probability
    for z in zeta: 
        probabilitys.append(np.exp(z)/total)
        print("Probability: ", np.exp(z)/total)


if __name__ == "__main__": # Run this testing code when layer file called directly 
    print("Hello, this is the direct file you are running from")
    input = Layer(5, 2)
    for neuron in input.neurons: 
        print(neuron)
        neuron.forwardActivation([2, 2])

    softmax(input)
    pass
 