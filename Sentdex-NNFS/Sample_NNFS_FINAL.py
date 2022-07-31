"""
Sample NNFS
"""

#Imports
from ossaudiodev import SNDCTL_DSP_SAMPLESIZE
from random import random
import numpy as np
import nnfs
from nnfs.datasets import vertical_data 

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
#Loss Class
class Loss:
    def calculate(self,output,y):
        sample_losses = self.forward(output,y)
        data_loss = np.mean(sample_losses)
        return data_loss
class Loss_CategoricalCrossentropy(Loss):
    def forward(self,y_pred,y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            corrected_confidences = y_pred_clipped[range(samples),y_true]
        
        elif len(y_true.shape)== 2:
            corrected_confidences =np.sum(y_pred_clipped*y_true,axis=1)
        
        negative_log_likelihoods = -np.log(corrected_confidences)
        return negative_log_likelihoods

#Sample Data Set
X,y = vertical_data(samples = 100, classes= 3)

#Setting Up Layers and Functions
dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3,3)
activation2 = Activation_Softmax()
loss_function = Loss_CategoricalCrossentropy()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

loss = loss_function.calculate(activation2.output,y)

#predefined sets to find best values of neurons
lowest_loss = 999999
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()


#iterations for training.
for iteration in range(100000):
    #setup weights and biases randomly 3 dimensional within 1 and -1
    dense1.weights = 0.05 * np.random.rand(2,3)
    dense1.biases = 0.05 * np.random.rand(1,3)
    dense2.weights = 0.05 * np.random.rand(3,3)
    dense2.biases = 0.05 * np.random.rand(1,3)

    #calculation in neurons
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    #calculation of loss y is samples results.
    loss = loss_function.calculate(activation2.output,y)

    #check predictions and accuracy
    predictions = np.argmax(activation2.output,axis=1)
    accuracy = np.mean(predictions==y)

    #Update best values of neurons
    if loss < lowest_loss:
        print('New set of weights found,iteration:',iteration,
                                        "loss:",loss,
                                        "acc:",accuracy)   
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss
    
print('Best Values Of Neurons')
print('Dense1 Weights and Biases')
print(best_dense1_weights,best_dense1_biases)
print('Dense2 Weights and Biases')
print(best_dense1_weights,best_dense1_biases)
print('Loss:', loss)




