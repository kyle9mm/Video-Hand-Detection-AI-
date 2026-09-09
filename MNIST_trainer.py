import numpy as np
import sigmoidNeuron
import layer
import neuralNetwork
from sklearn.datasets import fetch_openml

HIDDEN_LAYER1_COUNT =  16
HIDDEN_LAYER2_COUNT = 16
OUTPUT_LAYER_COUNT = 10
TRAINING_RATE1 = 0.11
TRAINING_RATE2 = 0.11


# Get all images to use as training
mnist = fetch_openml('mnist_784', version=1)
#This is my neural network to recognise hand written numbers
model1 = neuralNetwork.NeuralNetwork([HIDDEN_LAYER1_COUNT, HIDDEN_LAYER2_COUNT, OUTPUT_LAYER_COUNT])
model1.load("trained_model3.npz")
# for z in range(10): 
#     for i in range(4000): 
#         image = np.array(mnist.data.iloc[i])
#         image = image/255
#         target = mnist.target[i]
#         number_detector1.train(image, target, TRAINING_RATE1)
#         number_detector2.train(image, target, TRAINING_RATE2)
for z in range(3): 
    for i in range(70000):  
        image = np.array(mnist.data.iloc[i])
        image = image/255
        target = mnist.target[i]
        model1.train(image, target, TRAINING_RATE1)
    
print("Testing predictions of model with LEARNING_RATE1 = ", TRAINING_RATE1)
for i in range(60):
    test_image = np.array(mnist.data.iloc[i+10000]) / 255
    print("Target: ", mnist.target[i+10000], ", Prediction: ", model1.prediction(test_image))

accuracy_model1 = neuralNetwork.accuracy(model1, 60000, 1000, mnist.data, mnist.target)
print("Model one Learning Rate: ", TRAINING_RATE1, " amount correct: ", accuracy_model1)

model1.save("trained_model4")

