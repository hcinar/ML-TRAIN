#Imports
import numpy as np

np.random.seed(0)

#Make it dense in 1 and -1 
#Inputs
X = [[1,2,3,2.5],
     [3,1,3.2,2.1],
     [0.5,1,1,2.8]]      

"""
Number of inputs = n_inputs
number of neurons = n_neurons
"""
class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.10*np.random.rand(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights)+self.biases

layer1 = Layer_Dense(4,9)
layer2 = Layer_Dense(9,9)
layer3 = Layer_Dense(9,9)
layer4 = Layer_Dense(9,9)
layer5 = Layer_Dense(9,4)

layer1.forward(X)
layer2.forward(layer1.output)
layer3.forward(layer2.output)
layer4.forward(layer3.output)
layer5.forward(layer4.output)
print(layer5.output)
