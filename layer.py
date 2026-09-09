import numpy as np
import sigmoidNeuron

class Layer:
    def __init__(self, neuron_count, num_inputs):
        self.neuron_count = neuron_count
        self.forward = []
        self.z = []
        weights_list = []
        bias_list = []
        # Get all the weights and biases from the neurons and store them in matrixes 
        for i in range(neuron_count):
            neuron = sigmoidNeuron.SigmoidNeuron(num_inputs)
            weights_list.append(neuron.weights)
            bias_list.append(neuron.bias)
        self.weights_matrix = np.array(weights_list) # Make this into a numpy array 
        self.bias_matrix = np.array(bias_list) # Make this into a numpy array 

    def __str__(self):
        parts = []
        for i in range(len(self.weights_matrix)):
            value = self.forward[i] if len(self.forward) > 0 else 0
            parts.append(f"({float(value):.2f})")
        return " ".join(parts)
    
    def layer_forward(self, inputs): # Calculate forward value of all neurons in this layer (give how active they are)
        self.z = []
        self.forward = []
        # Calculate all the z values and the activation values 
        self.z = np.dot(np.array(self.weights_matrix), inputs) + np.array(self.bias_matrix)
        self.forward = 1 / (1 + np.exp(-self.z))

    def output_error(self, target_matrix): # Calculate error of the output of the network (How close it is to the correct output)
        return (self.forward - target_matrix) * (self.forward * (1 - self.forward))
    
    def hidden_layer_error(self, next_layer, next_layer_error): # Calculate error of the hidden layer
        propagated_error = np.dot(next_layer.weights_matrix.T, next_layer_error)
        hidden_error = propagated_error * (self.forward * (1 - self.forward))
        return hidden_error
    
    def layer_gradient(self, error, inputs): # Calculate gradient descent of weight and bias (How much does the weight and bias need to change)
        bias_gradient = error
        weight_gradient = np.outer(error, inputs) # Activations of last layer TIMES Error 
        return weight_gradient, bias_gradient 

    def update_layer(self, bias_gradient, weight_gradient, training_rate): # Change weight and bias values of the network so cost function is lower 
        self.bias_matrix -= bias_gradient * training_rate # move the biases in opposite direction of gradient by subtracting as it decreases cost function
        self.weights_matrix -= weight_gradient * training_rate # move the weights in opposite direction of gradient by subtracting as it decreases cost function


def softmax(self): # Use softmax to calculate the probability of each neuron being the correct answer 
    probabilitys = [] # Array to store probabilitys of neurons 
    # Calculate sum of all exponentials to the power of zeta 
    exp_values = np.exp(self.z) 
    total = sum(exp_values)
    # Go through each zeta value for each neuron and calculate its probability
    probabilitys = exp_values/total 
    return probabilitys

    

