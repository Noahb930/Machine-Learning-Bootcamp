import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_size, activation_func, loss_func, lr):
        self.weights =  np.random.randn(input_size, 1)  # Weights are a randomly initialized matrix with dimensions input_size x 1
        self.bias = np.random.randn(1,1) # The bias is a randomly initialized value
        self.activation_func = activation_func # The activation function determines the threshold at which the perceptron fires
        self.loss_func = loss_func # The loss function determines how the perceptron is penalized durring the training process
        self.loss_curve = np.array([]) # Set the loss curve to an empty array
        self.lr = lr # The learning rate of the network determines how large of a step it will take along the gradient

    def activation(self,inputs):
        if self.activation_func == "sigmoid": # Check if the perceptron has a sigmoid activation function
            return 1 / (1 + np.exp(-1*inputs)) # Pass the inputs through the sigmoid function
        elif self.activation_func == "relu": # Check if the perceptron has a ReLU activation function
            return np.max(inputs,0) # Pass the inputs through the ReLU function
    
    def activation_derivative(self,outputs): 
        if self.activation_func == "sigmoid": # Check if the perceptron has a sigmoid activation function
            return outputs * (1.0 - outputs) # Calculate the derivative of the sigmoid function
        elif self.activation_func == "relu": # Check if the perceptron has a ReLU activation function
            return outputs > 0 # Calculate the derivative of the activation function

    def loss(self, y_hat, y):
        if self.activation_func == "mean_squared_error": # Check if the perceptron has a mean squared error loss function
            return np.square(np.subtract(y_hat, y)).mean() # Calculate mean squared error
        elif self.activation_func == "binary_cross_entropy": # Check if the perceptron has a binary cross entropy loss function
            return np.mean(-1 * (y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))) # Calculate binary cross entropy
    
    def loss_derivative(self, y_hat, y):
        if self.activation_func == "mean_squared_error": # Check if the perceptron has a mean squared error loss function
            return 2 * np.subtract(y_hat, y) # Calculate the partial derivative of mean squared error with respect to y_hat
        elif self.activation_func == "binary_cross_entropy": # Check if the perceptron has a binary cross entropy loss function
            return (y_hat - y)/(y_hat * (1 - y_hat)) # Calculate the partial derivative of binary cross entropy with respect to y_hat

    def feedforward(self,inputs):
        self.x = inputs # x is equal to the input of the perceptron
        self.z = np.dot(self.x,self.weights) + self.bias # z = xw + b
        self.y_hat = self.activation(self.z) # Pass z into the activation function before returning the output
        return self.y_hat # Return the output of the perceptron

    def backprop(self, outputs, ground_truths):
        dLdy_hat = self.loss_derivative(outputs, ground_truths) # Calculate the partial derivative of loss with respect to y_hat
        dLdz = dLdy_hat * self.activation_derivative(self.y_hat) # Calculate the partial derivative of loss with respect to z
        dLdw = self.x.T.dot(dLdz) # Calculate the partial derivative of loss with respect to the weights
        dLdb = np.sum(dLdz,axis=0) # Calculate the partial derivative of loss with respect to the bias
        dLdx = dLdz.dot(self.weights.T) # Calculate the partial derivative of loss with respect to the perceptron's input
        self.weights -= self.lr * dLdw # Adjust the perceptron's weights by a fraction of the partial derivative dL/dw 
        self.bias -= self.lr * dLdb # Adjust the perceptron's bias by a fraction of the partial derivative dL/db
    
    def load(self):
        checkpoint = np.load("checkpoint.npz") # Load the file checkpoint.npz
        self.weights =  checkpoint["weights"] # Set the perceptron's weights to the weights contained in the checkpoint file
        self.bias =  checkpoint["bias"] # Set the perceptron's bias to the bias contained in the checkpoint file
        self.loss_curve = checkpoint["loss_curve"] # Recover the perceptron's loss curve from the checkpoint file
    
    def save(self):
        np.savez("checkpoint.npz", weights = self.weights, bias = self.bias) # Save the values of weights and biases to the file checkpoint.npz

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
        for epoch in range(epochs): # Conduct training for epochs iterations
            outputs = self.feedforward(inputs) # Feed the inputs into the perceptron and return the predicted value
            loss = self.loss(outputs, ground_truths) # Calculate the error between the predicted and correct values
            self.loss_curve = np.append(self.loss_curve, [loss])
            self.backprop(outputs, ground_truths) # Backpropagate the error through the perceptron
        self.save() # Save the trained model