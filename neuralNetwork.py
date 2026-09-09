import numpy as np
import sigmoidNeuron
import layer

class NeuralNetwork: 
    # Constructor takes a list holding numbers for the neuron count of each layer e.g [10, 16, 16, 4]
    def __init__(self, layers_neuroncount_list): 
        self.network = [] # Network list holds layer classes for each layer of the network
        # Create the first layer, as its special only having one input for intensity of each pixel
        last_count = 784 # first hidden layer will take all the pixel intensitys as an input (784 of them) 
        for count in layers_neuroncount_list: 
            self.network.append(layer.Layer(count, last_count))
            last_count = count
    
    def __str__(self):
        lines = []
        lines.append("INPUT: 784 inputs (image pixels)")
        lines.append("   |")
        lines.append("   v")

        for i, l in enumerate(self.network):
            is_output = (i == len(self.network) - 1)
            label = "OUTPUT LAYER" if is_output else f"HIDDEN LAYER {i + 1}"
            lines.append(f"{label}: {str(l)}")
            if not is_output:
                lines.append("   |")
                lines.append("   v")

        return "\n".join(lines)

    def forward(self, input): # Do one forward propogation of the network
        for layer_count in range(len(self.network)): # Keep count of which layer you are on 
            current_layer = self.network[layer_count]
            if layer_count == 0: # Compute forward with image pixels if its the first layer 
                current_layer.layer_forward(input)
                continue 
            current_layer.layer_forward(self.network[layer_count-1].forward) # Give input as the last layers forward activations 
    
    def target_matrix(self, target): # takes a target number e.g 5 andmakes it into a array of 0's and the correct number e.g [0 0 0 0 0 5 0 0 0 0]
        target = int(target)  # ensure it's an integer for comparison
        matrix = []
        for i in range(10):
            if i == target:
                matrix.append(1)
            else:
                matrix.append(0)
        return matrix

    def cost(self, target): # Get the cost function of the network using , which is how bad our current prediction is 
        print("computing cost")
        target_matrix = self.target_matrix(target) 
        last_layer = self.network[-1]
        return (1/last_layer.neuron_count)*np.square(target_matrix - last_layer.forward)

    def train(self, input_data, target, training_rate): # Trains model by propogating forward with image, computing error,
        self.forward(input_data)#                         the gradients and then updating weights and biases to decrease cost
        target_matrix_result = self.target_matrix(target)

        next_layer = None
        next_error = None

        for i in range(len(self.network) - 1, -1, -1):
            current_layer = self.network[i]

            # Determine this layer's error
            if i == len(self.network) - 1:
                error = current_layer.output_error(target_matrix_result)
            else:
                error = current_layer.hidden_layer_error(next_layer, next_error)

            # Determine what fed into this layer
            if i == 0:
                activation_inputs = input_data
            else:
                activation_inputs = self.network[i - 1].forward

            # Get gradients and update
            weight_gradient, bias_gradient = current_layer.layer_gradient(error, activation_inputs)
            current_layer.update_layer(bias_gradient, weight_gradient, training_rate)

            # Save this layer + its error for the next (earlier) iteration
            next_layer = current_layer
            next_error = error

    def prediction(self, image): # Give the predicted number on a input image
        self.forward(image)
        output = self.network[-1]
        predictions = layer.softmax(output)
        guess = None
        current_guess_probability = 0 
        # Loop through all the probabilitys and find the most likely one and set the number to guess
        for i in range(10): 
            if (predictions[i] > current_guess_probability): 
                guess = i 
                current_guess_probability = predictions[i]
        return guess
    
    def save(self, filepath):
        save_dict = {}
        for i, l in enumerate(self.network):
            save_dict[f"layer{i}_weights"] = l.weights_matrix
            save_dict[f"layer{i}_bias"] = l.bias_matrix
        np.savez(filepath, **save_dict)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        data = np.load(filepath)
        for i, l in enumerate(self.network):
            l.weights_matrix = data[f"layer{i}_weights"]
            l.bias_matrix = data[f"layer{i}_bias"]
        print(f"Model loaded from {filepath}")

def accuracy(self, start_index, num_images, mnist_data, mnist_target): # Runs model through MNIST Data base and gives accuracy of how many images cassified correct. 
    correct = 0
    for i in range(start_index, start_index + num_images):
        test_image = np.array(mnist_data.iloc[i]) / 255
        actual = int(mnist_target[i])
        predicted = self.prediction(test_image)
        if predicted == actual:
            correct += 1
    return (correct / num_images) * 100


