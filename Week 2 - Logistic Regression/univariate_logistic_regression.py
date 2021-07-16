import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class UnivariateLogisticRegression:
    def __init__(self, x, y):
        self.x = x # Save x data as an instance variable
        self.y = y # Save y data as an instance variable
        self.m = 0 # Start with a slope of 0
        self.b = np.mean(y) # Start by always guessing the mean of y

    def forward(self, x):
        self.z = x * self.m + self.b # z = mx + b
        y_hat = 1 / (1 + np.exp(-1 * self.z)) # y_hat = 1/(1 + e^-z))
        return y_hat

    def calculate_loss(self, y, y_hat):
        L = np.mean(-1 * (y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))) # Loss is equal to binary cross entropy
        return L

    def calculate_dLdy_hat(self, y, y_hat):
        dLdy_hat = -1 * (y / y_hat - (1 - y) / (1 - y_hat) ) # Calculate the partial derivative of the loss function with respect to y_hat
        return dLdy_hat

    def calculate_r_squared(self, y, y_hat):
        y_bar = np.mean(y) # Calculate the mean of y
        rss = np.sum((y - y_hat) ** 2) # Calculate the residual sum of squares
        tss = np.sum((y - y_bar) ** 2) # Calculate the total sum of squares
        r_squared = 1 - rss/tss # Calculate the pearson correlation coefficient
        return r_squared

    def backwards(self, dLdy_hat, x):
        dLdz = dLdy_hat * np.exp(-1 * self.z)/((1 + np.exp(-1 * self.z)) ** 2)
        dLdm = dLdz * x # Calculate the partial derivative of the loss function with respect to m
        dLdb = dLdz # Calculate the partial derivative of the loss function with respect to b
        self.m -= 0.5 * np.mean(dLdm) # Update the value of m
        self.b -= 0.5 * np.mean(dLdb) # Update the value of b
    
    def update_fig(self, fig, ax, x, y, y_hat):
        plt.title("Global Warming and Carbon Dioxide") # Add figure title
        plt.xlabel("Atmospheric Carbon Dioxide (‱)") # Add x-axis label
        plt.ylabel("Temperature (°C)") # Add y-axis label
        plt.scatter(x, y) # Plot the datapoints
        x_bounds = np.array([np.min(x),np.max(x)]) # get maximum and minimum x-coordinate
        plt.plot(x_bounds,x_bounds*self.m+self.b, color='black') # Plot the line of best fit
        plt.text(0.025, 0.95, f'y = {np.round(self.m,3)}x + {np.round(self.b,3)}', fontsize=8, transform = ax.transAxes) # Write the equation for the line of best fit
        plt.text(0.025, 0.9, f'r^2 = {np.round(self.calculate_r_squared(y, y_hat),3)}', fontsize=8, transform = ax.transAxes) # Write the value for r^2
        plt.draw()  # Draw the figure
        plt.pause(1e-4) # Wait 1*10^-4 seconds
        fig.clear() # Clear the figure

    def train(self, epochs):
        # fig, ax = plt.subplots() # Create a new figure and set of axis
        for epoch in range(epochs): # Train for chosen number of epochs
            y_hat = self.forward(self.x) # Calculate the predicted value of x
            print(self.calculate_loss(self.y,y_hat))
            dLdy_hat = self.calculate_dLdy_hat(self.y, y_hat) # Calculate the derivitive of our loss function with respect to y_hat
            self.backwards(dLdy_hat, self.x) # Backpropagate the error
            # self.update_fig(fig, ax, self.x, self.y, y_hat) # Update the figure
        # plt.show() # Show the figure for the trained model