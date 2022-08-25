#This is RUKA
#AI itself.
AI = "RUKA"
version = "v0"

#Imports
from random import random
import numpy as np
from numpy import zeros
import pandas as pd


#SYMBOL
symbol = "BTCUSDT"

#Data Import Paths 
data_import_paths= [r"/home/hus/GIT/ML-TRAIN/Neural Network for Finance/"+symbol+"D"+'CHG'+'.xlsx']

#Data Export Paths
data_export_paths = [r"/home/hus/GIT/ML-TRAIN/Neural Network for Finance/"+AI+"_"+'v0'+'.xlsx']

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
        
        #4 Column
        elif len(y_true.shape)== 2:
            corrected_confidences =np.sum(y_pred_clipped*y_true,axis=None)
        
        negative_log_likelihoods = -np.log(corrected_confidences)
        return negative_log_likelihoods

#Remove OpenTime and Nulls
def dataSetCorrection(dframe):
    del dframe['Open time']
    dframe = dframe[dframe.WOPEN.notnull()]
    return dframe


def TrainLuka(data_set,to_row):
    #correct dataset
    data_set = dataSetCorrection(data_set)
    print("data_set\n",data_set)

    #get training data
    training_data = data_set[:to_row-1]
    print("training_data\n",training_data)

    #get training results
    training_results_data = data_set[1:to_row]
    del training_results_data['WOPEN']
    del training_results_data['WHIGH']
    del training_results_data['WCLOSE']
    del training_results_data['WLOW']
    print("training_results_data\n",training_results_data)

    # X =   *Train set
    # y =   *Correct answers.
    X = training_data
    y = training_results_data

    #Normalize Data!

    

    #Setting Up Layers and Functions
    dense1 = Layer_Dense(8,99)
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(99,4)

    activation2 = Activation_Softmax()
    loss_function = Loss_CategoricalCrossentropy()

    #Check here!
    dense1.forward(X[:1])
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    loss = loss_function.calculate(activation2.output,y[:1])

    #predefined sets to find best values of neurons
    lowest_loss = 999999
    best_dense1_weights = dense1.weights.copy()
    best_dense1_biases = dense1.biases.copy()
    best_dense2_weights = dense2.weights.copy()
    best_dense2_biases = dense2.biases.copy()


    #iterations for training.
    for iteration in range(100000):
        #setup weights and biases randomly
        #SETUP could be wrong rn.
        dense1.weights = 0.05 * np.random.rand(8,99)
        dense1.biases = 0.05 * np.random.rand(1,99)
        dense2.weights = 0.05 * np.random.rand(99,4)
        dense2.biases = 0.05 * np.random.rand(1,4)

        #calculation in neurons
        #Calculation rn is wrong.
        dense1.forward(X[:2])
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        activation2.forward(dense2.output)

        #calculation of loss y is samples results.
        #accuarry is wrong but works :)
        loss = loss_function.calculate(activation2.output,y[:2])

        #check predictions and accuracy
        #accuarry is wrong but works :)
        predictions = np.argmax(activation2.output,axis=None)
        accuracy = np.mean(predictions==y[:2])

        #Update best values of neurons
        if loss < lowest_loss:
            print('New set of weights found,iteration:\n',iteration,
                                            "loss:\n",loss,
                                            "acc:\n",accuracy)   
            best_dense1_weights = dense1.weights.copy()
            best_dense1_biases = dense1.biases.copy()
            best_dense2_weights = dense2.weights.copy()
            best_dense2_biases = dense2.biases.copy()
            lowest_loss = loss
    
    print('Best Values Of Neurons')
    print('Dense1 Weights and Biases')
    print(best_dense1_weights,best_dense1_biases)
    print('Dense2 Weights and Biases')
    print(best_dense2_weights,best_dense2_biases)
    print('Loss:', loss)



if __name__ == "__main__":
    #Read excel from paths
    for path_index,i_path in enumerate(data_import_paths):

        #Read excel from path
        df = pd.read_excel(i_path)

        #Train luka data set
        row_count_for_training = 1000
        TrainLuka(df,row_count_for_training)

        #Will save file with train results and RUKA's Settings :).
        #Export to excel file.
        e_path = data_export_paths[int(path_index)]
        #df.to_excel(e_path,index=False)

        #Give info when completed.
        print(i_path,"\n Exported Into-----> ",e_path)
