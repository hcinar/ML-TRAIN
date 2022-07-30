#Imports
import numpy as np
import nnfs 
from nnfs.datasets import spiral_data

#Neural Network From Scratch Initialization
nnfs.init()


#Class for Neuron Calculation
class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.10*np.random.rand(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights)+self.biases

#Class for Activation of Inputs
class Activation_ReLU:
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)


#Expo and Normalize
class Activation_Softmax:
    def forward(self,inputs):
        exp_values = np.exp(inputs - np.max(inputs,
                                            axis = 1, 
                                            keepdims=True))
        probabilities = exp_values / np.sum(exp_values ,
                                            axis = 1,
                                            keepdims=True)
        self.output = probabilities

#Sample Data Set
X,y = spiral_data(samples = 100, classes= 3)


#Setting Up Layers
dense1 = Layer_Dense(2,7)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(7,3)
activation2 = Activation_Softmax()

#DO Calculation
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])