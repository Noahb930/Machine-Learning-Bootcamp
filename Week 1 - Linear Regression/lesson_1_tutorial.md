# Part 1 - Linear Regression
## Introduction
When we talk about machine learning, we often think of Neural Networks and other complex model architectures. But even something as commonplace as the linear regression or line of best fit is an example of machine learning. When finding the line of best fit, we "learn" to fit a line to a set of incomplete data points. We then use this line to extrapolate the value of points that we haven't seen before. That's machine learning. While I'm sure all of you have had to run a linear regression at least once, we don't often stop to think about what happens behind the scenes after we press calculate. This tutorial will cover the math, logic, and implementation for programming a linear regression from scratch in Python.
## Importing Modules
This tutorial will use three python modules, NumPy for linear algebra, Pandas for data manipulation, and Matplotlib for graphing. Make sure these three libraries are installed and then import them at the beginning of your file.
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```
## The Forward Pass
In order for our model to learn, we start by making a random guess. Lets declare a class called Model with instance variables m and b set to 0. Here m will represent the slope of our line and b will represent the y-intercept.
```python
class Model:
    def __init__(self, guess):
        self.m = 0 # Start with a slope of 0
        self.b = guess # Start with chosen y-intercept
```
Now that we've initialized our m and b variables, we can write the forward method for our class which makes a prediction by using the equation for a line:

![image](https://cdn-images-1.medium.com/max/1600/1*i45CBQ9J51Wmf-7nywwwZw.png)

```python
def forward(self, x):
    y_hat = x * self.m + self.b # y_hat = mx + b
    return y_hat
```
Once we've made a guess, we need to calculate the error or loss so that we can make adjustments to our model's parameters. Linear regressions calculate loss using the mean squared error loss function. Here ŷ represents our models prediction and y represents the actual y value for a datapoint

![image](https://cdn-images-1.medium.com/max/1600/1*mZ0BXafhAyUqSExPSA2-OA.png)

```python
def calculate_loss(self, y, y_hat):
    L = np.mean((y - y_hat) ** 2) # Loss is equal to mean squared error
    return L
```

## Gradient Descent
Our linear regression is going to learn using an algorithm called gradient descent. In gradient decent, we calculate the partial derivative of each model parameter with respect to the loss. We can think of the partial derivitive as a vector pointing in the direction we need to travel. To travel along the gradient, we simply subtract the partial derivative of each parameter from the parameter itself. As we get closer to the absolute minimum, we need to make smaller and smaller adjustments.
## The Backward Pass
In order to calculate the partial derivatives ∂L/∂m and ∂L/∂b, we first need to calculate ∂L/∂ŷ.

![image](https://cdn-images-1.medium.com/max/1600/1*mZ0BXafhAyUqSExPSA2-OA.png)
```python
def calculate_dLdy_hat(self, y, y_hat):
    dLdy_hat = (y_hat - y) * 2 # Calculate the partial derivative of the loss function with respect to y_hat
    return dLdy_hat
```
Now using the chain rule and our derivitive rules, we can calculate the partial derivative of the loss with respect to each parameter.
```python
def backwards(self, dLdy_hat, x):
    dLdm = dLdy_hat * x # Calculate the partial derivative of the loss function with respect to m
    dLdb = dLdy_hat # Calculate the partial derivative of the loss function with respect to b
```
We can then make adjustments to each model parameter by subtracting a portion of the corresponding partial derivtive.
We multiply the partial derivitive by 0.1 to prevent the model from making too large of an adjustment. Here, 0.1 is an arbitrary value called the learning rate. The learning rate of a model is a hyperparameter - a parameter that is not altered during training
```python
def backwards(self, dLdy, x):
    ...
    self.m -= 0.8 * np.mean(dLdm) # Update the value of m
    self.b -= 0.8 * np.mean(dLdb) # Update the value of b
```
## Putting it All Together
Now we have all the individual parts that we need for one complete epoch (iteration) of training.
```python
def train(self, x, y):
    for epoch in range(5000): # Train for 5000 epochs
        y_hat = self.forward(x) # Calculate the predicted value of x
        dLdy_hat = self.calculate_dLdy_hat(y, y_hat) # Calculate the derivitive of our loss function with respect to y_hat
        self.backwards(dLdy_hat, x) # Backpropagate the error
```
Lets now use our main function to load our data and train the model on it
```python
if __name__ == "__main__":
    df = pd.read_csv('data.csv') # Create a dataframe from the .csv file
    model = Model(np.mean(df['Temperature'])) # Create an instance of the Model class
    model.train(df['CO2 PPM']/1000,df['Temperature']) # Train the model on our data
```
## Graphing Progress
To better visualize the progress of our model, we can also graph the linear regression.
```python
def update_fig(self, fig, x, y_hat):
    plt.scatter(x, y) # Plot the datapoints
    x_bounds = np.array([np.min(x),np.max(x)]) # get maximum and minimum x-coordinate
    plt.plot(x_bounds,x_bounds*self.m+self.b, color='black') # Plot the line of best fit
    plt.text(0.025, 0.95, f'y = {np.round(self.m,3)}x + {np.round(self.b,3)}', fontsize=8, transform = ax.transAxes) # Write the equation for the line of best fit
    plt.draw()  # Draw the figure
    plt.pause(1e-4) # Wait 1*10^-4 seconds
    fig.clear() # Clear the figure
def train(self, x, y_hat):
    fig = plt.figure()
    for epoch in range(5000):
        ...
        self.update_fig(fig, x, y_hat)
```
## Calculating The Fit
While we previously calculated the mean squared error of the regression, this alone doesn't give us enough information about the strength of the correlation. To do so, we need to know three quantities, the Total Sum of Squares (TSS), Residual Sum of Squares RSS, and Explainable Sum of Squares (ESS). 
* TSS indicates the variation of a sample from its mean
* RSS indicates the variation of a prediction from the sample
* ESS indicates the variation of a prediction from the mean of the sample
<div width="100%" style="display:flex;justify-content:space-around">
    <img src="https://render.githubusercontent.com/render/math?math=TSS%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7B(y_i-%5Cbar%7By%7D)%5E2%7D">
    <img src="https://render.githubusercontent.com/render/math?math=RSS%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7B(y_i-%5Chat%7By_i%7D)%5E2%7D">
    <img src="https://render.githubusercontent.com/render/math?math=ESS%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7B(%5Chat%7By_i%7D-%5Cbar%7By%7D)%5E2%7D">
</div>
Since TSS = RSS + ESS, we can then use these quantities to calculate the pearson correlation coefficient according to the following equation:
<div align="center">
    <img src="https://render.githubusercontent.com/render/math?math=r%5E2%20%3D%20%5Cfrac%7BESS%7D%7BTSS%7D%20%3D%201%20-%20%5Cfrac%7BRSS%7D%7BTSS%7D">
</div>
In other words, the value of r squared is equal to the proportion of the sample's variation that can be explained.

```python
def calculate_r_squared(self, y, y_hat):
    y_bar = np.mean(y)
    rss = np.sum((y - y_hat) ** 2)
    tss = np.sum((y - y_bar) ** 2)
    r_squared = 1 - rss/tss
    return r_squared
```
We can now plot the value of r squared beneath the equation of the trend line
```python
def update_fig(self, fig, x, y_hat):
    ...
    plt.text(0.025, 0.95, f'y = {np.round(self.m,3)}x + {np.round(self.b,3)}', fontsize=8, transform = ax.transAxes) # Write the equation for the line of best fit
    plt.text(0.025, 0.9, f'r^2 = {np.round(self.calculate_r_squared(y, y_hat),3)}', fontsize=8, transform = ax.transAxes) # Write the value for r^2
    ...
```