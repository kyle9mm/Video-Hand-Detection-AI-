import numpy as np
import sigmoidNeuron

class Layer:
    def __init__(self, neuron_count, num_inputs):
        self.neurons = []
        self.z = []
        self.forward = []
        # Create the layer of neurons with random weights and biases 
        for i in range(neuron_count): 
            self.neurons.append(sigmoidNeuron.SigmoidNeuron(num_inputs))

    def __str__(self):
        parts = []
        for i in range(len(self.neurons)):
            value = self.forward[i] if i < len(self.forward) else 0
            parts.append(f"({float(value):.2f})")
        return " ".join(parts)
    
    def layer_forward(self, inputs): # Calculate value of this neuron (How active it is), self - instance of sigmoid class, inputs - np array of input values into the neuron
        self.z = []
        self.layer = []
        print("Hello, getting the output for you")
        # Calculate all the z values and the activation values 
        for neuron in self.neurons:
            z_value = np.dot(neuron.weights, inputs)
            self.z.append(z_value)
            self.forward.append(1 / (1 + np.exp(-z_value))) # Calculate the sigmoid value of the neuron which is its activation


  
def softmax(layer): # Use softmax to calculate the probability of each neuron as being the most likely correct answer
    probabilitys = []
    # Calculate sum of all exponentials to the power of zeta 
    exp_values = np.exp(layer.z) 
    total = sum(exp_values)
    # Go through each zeta value for each neuron and calculate its probability
    for z in layer.z: 
        probabilitys.append(np.exp(z)/total)
        print("Probability: ", np.exp(z)/total)


if __name__ == "__main__": # Run this testing code when layer file called directly 
    print("Hello, this is the direct file you are running from")
    input = Layer(5, 2)
    input.layer_forward([2, 2])

    softmax(input)
    pass
 