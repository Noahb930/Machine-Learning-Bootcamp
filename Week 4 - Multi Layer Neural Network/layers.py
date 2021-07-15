import numpy as np
import os
import itertools

class Layer:
    
    counter = itertools.count() #Package used to create unique incrementing layer id

    def activation(self,func,deriv,x):  #Defining the activation function
        if not deriv :
            if func == self.func: #Allows activation function to alternate but is by default sigmoid
                return 1 / (1 + np.exp(-x)) #Sigmoid formula
            elif func == "relu": #Relu is a seprate activation function
                if x > 0:
                    return x
                else:
    	            return 0
        else: #If derivative is true
            if func == self.func:
              return x * (1.0 - x) #Derivative of Sigmoid
            elif func == "relu": #Derivative of Relu
                if x>0:
                    return 1
                else:
                    return 0
    def save(self):
        np.savez(os.getcwd()+ "/layers/layer"+str(self.id)+".npz",weights =self.weights, bias=self.bias) #Saves file 

    def load(self): #Loads file
        self.weights = np.load(os.getcwd()+ "/layers/layer"+str(self.id)+".npz")["weights"] #Initializes weights from file
        self.bias = np.load(os.getcwd()+ "/layers/layer"+str(self.id)+".npz")["bias"] #Initializes biases from file

class FullyConnectedLayer(Layer):
     def __init__(self, func, hidden_size, output_size): #Layer definition with parameters
        self.id = next(self.counter) #Unique ID of layer
        self.func = func #Activation Function
        self.weights = np.random.randn(hidden_size, output_size) #Randomly initializing weights based on size parameters
        self.bias = np.random.randn(1,output_size) #Randomly initilaizing biases based on output size parameter

     def feed_forward(self,input_layer):
        self.input_layer = input_layer #Initialzing the input
        self.z = np.dot(input_layer, self.weights) + self.bias #Performing layer operations (dot product of input and weights then adding the bias) 
        self.output = self.activation(self.func, False, self.z) #Applying activation function
        return self.output #Returning output
    
     def back_prop(self,prev_layer_delta, lr):
        self.d_z = prev_layer_delta*(self.activation(self.func,True, self.output)) #Layer output delta
        self.d_w = self.input_layer.T.dot(self.d_z) #Weights delta 
       
        self.d_x = self.d_z.dot(self.weights.T) #Input layer delta
        self.d_b = np.sum(self.d_z,axis=0) #Bias delta
        
        self.bias = self.bias - (self.d_b*lr) #Update bias
        self.weights = self.weights - (self.d_w*lr) #Update weights
        return self.d_x


        








