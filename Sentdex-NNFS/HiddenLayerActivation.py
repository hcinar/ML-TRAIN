#Imports
import numpy as np
import nnfs 
from nnfs.datasets import spiral_data

#Neural Network From Scratch Initialization
nnfs.init()

#Input Data Sets
#2 Inputs Batch size 100
X, y = spiral_data(100, 3)


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

#Layer with 2 Inputs and 5 Neuron
layer1 = Layer_Dense(2,5)
#Activation of 5 Neuron
activation1 = Activation_ReLU()

#Input Batch into Layer1
layer1.forward(X)
#Activate Layer1 Output
activation1.forward(layer1.output)
print(activation1.output)