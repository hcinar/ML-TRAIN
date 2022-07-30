"""
Basic Neuron
"""


#Inputs for Neuron
inputs = [1.2,5.1,2.1]
#Neuron weights
weights=[3.1,2.1,8.7]
#bias of 
bias = 3

output = inputs[0]* weights[0] + inputs[1] * weights[1] + inputs[2]* weights[2] + bias
print(output)