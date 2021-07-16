import numpy as np
import matplotlib.pyplot as plt

class Network:
    def __init__(self, name, layers, loss_func, lr):
        self.name = name # Initializing name
        self.layers = layers # Initializing layers
        self.loss_func = loss_func # The loss function determines how the perceptron is penalized durring the training process
        self.loss_curve = np.array([]) # Set the loss curve to an empty array
        self.lr = lr # Initializing learning rate
    
    def loss(self, y_hat, y):
        if self.loss_func == "mean_squared_error": # Check if the perceptron has a mean squared error loss function
            return np.square(np.subtract(y_hat, y)).mean() # Calculate mean squared error
        elif self.loss_func == "binary_cross_entropy": # Check if the perceptron has a binary cross entropy loss function
            return np.mean(-1 * (y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))) # Calculate binary cross entropy
    
    def loss_derivative(self, y_hat, y):
        if self.loss_func == "mean_squared_error": # Check if the perceptron has a mean squared error loss function
            return 2 * np.subtract(y_hat, y) # Calculate the partial derivative of mean squared error with respect to y_hat
        elif self.loss_func == "binary_cross_entropy": # Check if the perceptron has a binary cross entropy loss function
            return (y_hat - y)/(y_hat * (1 - y_hat)) # Calculate the partial derivative of binary cross entropy with respect to y_hat

    def save(self):
        np.savez(f"checkpoint/learning_curves.npz", loss = self.loss_curve) # Saves file
        for layer in self.layers:
            layer.save() # Saves each layer through layer class save method

    def load(self):
        self.loss_curve = np.load("checkpoint/learning_curves.npz")["loss"] # Load the loss curve from checkpoint/learning_curves.npz
        for layer in self.layers:
            layer.load() # Loads each layer through layer class load method

    def feedforward(self,inputs):
        signal = inputs
        for layer in self.layers: # Iterates through the list of layers
            signal = layer.feedforward(signal) # Passes through the layer through the layer class feed forward method    
        return signal
    
    def backprop(self, outputs, ground_truths):
        gradient = self.loss_derivative(outputs, ground_truths)
        for layer in reversed(self.layers):
           gradient = layer.backprop(gradient, self.lr) # Running each layer though the layer backpropogation function

    def graph(self, window):
        moving_average_loss = [] # Inititalize the moving average as an empty array
        for epoch in range(self.loss_curve.shape[0] - window): # The average will be taken epochs - window times
            moving_average_loss.append(np.mean(self.loss_curve[epoch:epoch+window])) # Append the average over a given window of epochs
        plt.plot(np.arange(len(moving_average_loss)),moving_average_loss) # Plot the moving average over epochs
        plt.xlabel('Epochs') # Label the x-axis
        plt.ylabel('Loss') # Label the y-axis
        plt.title("Training of a Perceptron") # Title the graph
        plt.show() # Show the graph

    def train(self, inputs, ground_truths, epochs):
        for epoch in range(epochs): # Iterates through epochs
            outputs = self.feedforward(inputs) # Calls feed forward
            loss = self.loss(outputs, ground_truths) # Calculate the error between the predicted and correct values
            self.loss_curve = np.append(self.loss_curve, [loss])
            self.backprop(outputs, ground_truths) # Calls back propogation
        self.save() # Saves network