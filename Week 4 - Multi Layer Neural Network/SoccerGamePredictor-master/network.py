import numpy as np


class Network:
    
    def __init__(self,n,l,lr):
        self.name = n #Initializing name
        self.layers = l #Initializing layers
        self.learning_R = lr #Initializing learning rate
        

    def feed_forward(self,input_layer):
        for layer in self.layers: #Iterates through the list of layers
            input_layer = layer.feed_forward(input_layer) #Passes through the layer through the layer class feed forward method    
        self.output_layer = input_layer #Sets output of this layer to input of the next
        return self.output_layer #Returns output

    def loss(self, output_layer, ground_truth):
        return np.square(np.subtract(output_layer, ground_truth)).mean() #Loss function
    
    def back_prop(self, ground_truth, output_layer):
        loss_derivatives = 2*(output_layer-ground_truth) #Deriv of loss function
        last_layer_delta = loss_derivatives #Last layer in network delta
        for layer in reversed(self.layers):
           last_layer_delta = layer.back_prop(last_layer_delta,self.learning_R) #Running each layer though the layer backpropogation function
        
    def save(self):
        for layer in self.layers:
            layer.save() #Saves each layer through layer class save method
    def load(self):
        for layer in self.layers:
            layer.load() #Loads each layer through layer class load method

    def train(self, input_layers, ground_truth, number_of_epochs):
        for epochs in range(number_of_epochs): #Iterates through epochs
            self.feed_forward(input_layers) #Calls feed forward
            self.back_prop(ground_truth, self.output_layer) #Calls back propogation
            if epochs%100 == 0: 
                print(self.loss(ground_truth, self.output_layer)) #Prints loss
        self.save() #Saves network
        

