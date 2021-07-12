from network import Network
from layers import FullyConnectedLayer
import numpy as np
import pandas as pd

data = pd.read_csv("SoccerData.txt", header = None, sep='\t')
ground_truth = data.iloc[:,8:]
data = data.drop(data.columns[[8, 9]], axis=1)
layers = []
layers.append(FullyConnectedLayer("relu",8,16))
layers.append(FullyConnectedLayer("relu",16,32))   
layers.append(FullyConnectedLayer("relu",32,64))   
layers.append(FullyConnectedLayer("relu",64,32))   
layers.append(FullyConnectedLayer("relu",32,16))
layers.append(FullyConnectedLayer("relu",16,8))
layers.append(FullyConnectedLayer("sigmoid",8,2))


network = Network("Test1",layers, 0.01)
network.load()
print(network.loss(network.feed_forward(data.to_numpy()),ground_truth.to_numpy()))

network.train(data.to_numpy(), ground_truth.to_numpy(), 5000)





