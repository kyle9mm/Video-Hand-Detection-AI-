import numpy as np
import sigmoidNeuron

class Layer:
    def __init__(self, neuron_count, num_inputs):
        #self.neurons = []
        self.neuron_count = neuron_count
        self.weights_matrix = []
        self.bias_matrix = []
        self.z = []
        self.forward = []
        # Create the layer of neurons with random weights and biases 
        for i in range(neuron_count): 
            neuron = sigmoidNeuron.SigmoidNeuron(num_inputs)
            #self.neurons.append(sigmoidNeuron.SigmoidNeuron(num_inputs))
            self.weights_matrix.append(neuron.weights)
            self.bias_matrix.append(neuron.bias)

    def __str__(self):
        parts = []
        for i in range(len(self.weights_matrix)):
            value = self.forward[i] if len(self.forward) > 0 else 0
            parts.append(f"({float(value):.2f})")
        return " ".join(parts)
    
    def layer_forward(self, inputs): # Calculate value of this neuron (How active it is), self - instance of sigmoid class, inputs - np array of input values into the neuron
        self.z = []
        self.forward = []
        print("Hello, getting the output for you")
        # Calculate all the z values and the activation values 
        self.z = np.dot(np.array(self.weights_matrix), inputs) + np.array(self.bias_matrix)
        self.forward = 1 / (1 + np.exp(-self.z))

    def error(self, target_matrix): 
        return (self.forward - target_matrix) * (self.forward * (1 - self.forward))
    
    def layer_gradient(self, error, inputs): 
        bias_gradient = error
        weight_gradient = np.outer(error, inputs) # Activations of last layer TIMES Error 
        return weight_gradient, bias_gradient 
    

  
def softmax(layer): # Use softmax to calculate the probability of each neuron as being the most likely correct answer
    probabilitys = []
    # Calculate sum of all exponentials to the power of zeta 
    exp_values = np.exp(layer.z) 
    total = sum(exp_values)
    # Go through each zeta value for each neuron and calculate its probability
    probabilitys = exp_values/total 
    #for p in probabilitys: 
    #    print("Probability: ", p)
    return probabilitys

    




if __name__ == "__main__": # Run this testing code when layer file called directly 
    print("Hello, this is the direct file you are running from")
    input = Layer(5, 2)
    input.layer_forward([2, 2])

    softmax(input)
    pass
 