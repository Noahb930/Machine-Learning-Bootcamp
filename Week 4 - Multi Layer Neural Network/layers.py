import numpy as np
import os
import itertools

class Layer:
    counter = 0
    
    def __init__(self, activation_func, hidden_size, output_size): # Layer definition with parameters
        self.id = self.counter # Unique ID of layer
        self.counter += 1
        self.activation_func = activation_func # Activation Function
        self.weights = np.random.randn(hidden_size, output_size) # Randomly initializing weights based on size parameters
        self.bias = np.random.randn(1,output_size) # Randomly initilaizing biases based on output size parameter
    
    def activation(self, inputs):
        if self.activation_func == "sigmoid": # Check if the perceptron has a sigmoid activation function
            return 1 / (1 + np.exp( -1 * inputs)) # Pass the inputs through the sigmoid function
        elif self.activation_func == "relu": # Check if the perceptron has a ReLU activation function
            return np.max(inputs, 0) # Pass the inputs through the ReLU function
    
    def activation_derivative(self, outputs): 
        if self.activation_func == "sigmoid": # Check if the perceptron has a sigmoid activation function
            return outputs * (1.0 - outputs) # Calculate the derivative of the sigmoid function
        elif self.activation_func == "relu": # Check if the perceptron has a ReLU activation function
            return outputs > 0 # Calculate the derivative of the activation function

    def save(self):
        np.savez(f"checkpoint/layer{str(self.id)}.npz", weights = self.weights, bias = self.bias) # Saves file 

    def load(self): # Loads file
        checkpoint = np.load(f"checkpoint/layer{str(self.id)}.npz")
        self.weights = checkpoint["weights"] # Initializes weights from file
        self.bias = checkpoint["bias"] # Initializes biases from file

    def feedforward(self, inputs):
        self.x = inputs #Initialzing the input
        self.z = np.dot(self.x, self.weights) + self.bias # Performing layer operations (dot product of input and weights then adding the bias) 
        self.y_hat = self.activation(self.z) # Applying activation function
        return self.y_hat # Returning output
    
    def backprop(self,gradient, lr):
        dLdz = gradient * (self.activation_derivative(self.y_hat)) # Layer output delta
        dLdw = self.x.T.dot(dLdz) # Weights delta 
        dLdb = np.sum(dLdz, axis = 0) # Bias delta
        dLdx = dLdz.dot(self.weights.T) # Input layer delta
        self.bias -= dLdb * lr # Update bias
        self.weights -= dLdw * lr # Update weights
        return dLdx


        








