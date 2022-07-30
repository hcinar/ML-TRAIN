"""
The Dot Product Numpy
Modelling 3 Neuron
"""
#Imports
import numpy as np

#Inputs
inputs = [1,2,3,2.5]      

#Weights of Inputs
weights=[[0.2,0.8,-0.5,1.0],
         [0.5,-0.91,0.26,-0.5],
         [0.2,0.8,-0.5,1.0]]

#Biases of neurons
biases = [2,3,1]

#Make Calculation with np.dot()
output = np.dot(weights,inputs)+biases
print(output)



"""
We make calculation with loops Basics are in Neuron.py

layer_outputs = [] #Outputs of current layer

#Neuron Calculations
for neuron_weights, neuron_bias in zip(weights,biases):
    neuron_output = 0 #output of given neuron
    for n_input, weight in zip(inputs,neuron_weights):
        neuron_output +=n_input*weight
    neuron_output+=neuron_bias 
    layer_outputs.append(neuron_output)

print(layer_outputs)



#Input weights 
weights=[[0.2,0.8,-0.5,1.0],
         [0.5,-0.91,0.26,-0.5],
         [0.2,0.8,-0.5,1.0]]

#biases of neurons
biases = [2,3,1]


"""
