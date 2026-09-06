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
        print("Hello")


if __name__ == "__main__": # Run this testing code when layer file called directly 
    print("Hello, this is the direct file you are running from")
    
    pass
 