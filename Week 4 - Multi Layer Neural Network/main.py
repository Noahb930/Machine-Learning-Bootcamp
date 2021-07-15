from network import Network
from layers import FullyConnectedLayer
import numpy as np
import pandas as pd

data = pd.read_csv("SoccerData.txt", header = None, sep='\t') #Reading the data
ground_truth = data.iloc[:,8:] #Setting the ground truth from the data frame
data = data.drop(data.columns[[8, 9]], axis=1) # Data cleaning
layers = []
layers.append(FullyConnectedLayer("relu",8,16)) #Adding layer with relu activation
layers.append(FullyConnectedLayer("relu",16,32)) #Adding layer with relu activation
layers.append(FullyConnectedLayer("relu",32,64)) #Adding layer with relu activation
layers.append(FullyConnectedLayer("relu",64,32)) #Adding layer with relu activation
layers.append(FullyConnectedLayer("relu",32,16)) #Adding layer with relu activation
layers.append(FullyConnectedLayer("relu",16,8)) #Adding layer with relu activation
layers.append(FullyConnectedLayer("sigmoid",8,2)) #Adding final layer with sigmoid to normalize outputs between 0-1


network = Network("Test1",layers, 0.01) #Declaring network
network.load() #Loading network
network.train(data.to_numpy(), ground_truth.to_numpy(), 5000) #Training Network






