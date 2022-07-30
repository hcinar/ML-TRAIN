"""
Batch Layer Object
"""

#Imports
import numpy as np

#Inputs
inputs = [[1,2,3,2.5],
          [3,1,3.2,2.1],
          [0.5,1,1,2.8]]      

#Weights of Inputs
weights=[[0.2,0.8,-0.5,1.0],
         [0.5,-0.91,0.26,-0.5],
         [0.2,0.8,-0.5,1.0]]

#Biases of neurons
biases = [2,3,1]

#Weights of Inputs
weights2=[[0.2,0.8,-3.5],
         [0.5,-1.91,0.26],
         [1.2,0.8,-0.5]]

#Biases of neurons
biases2 = [1,2,1]


#Make Calculation with np.dot()
#Layer1's outputs is become layer2's inputs

# (3 , 4) * (4 , 3) = ( 3 , 3 )
layer1_outputs = np.dot(inputs,np.array(weights).T)+biases
# (3 , 3) * (3 , 3) = ( 3 , 3 )
layer2_outputs = np.dot(layer1_outputs,np.array(weights2).T)+biases
print(layer2_outputs)



"""
Inputs (3,4)
Weights (3,4)
Weights Tranpose (4,3)

I X W cannot made.

I X W.Transpose can made.

"""