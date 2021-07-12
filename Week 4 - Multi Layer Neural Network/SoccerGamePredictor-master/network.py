import numpy as np


class Network:
    
    def __init__(self,n,l,lr):
        self.name = n
        self.layers = l
        self.learning_R = lr
        

    def feed_forward(self,input_layer):

        for layer in self.layers:
            input_layer = layer.feed_forward(input_layer)        
        self.output_layer = input_layer
        return self.output_layer

    def loss(self, output_layer, ground_truth):
        return np.square(np.subtract(output_layer, ground_truth)).mean() 
    
    def back_prop(self, ground_truth, output_layer):
        loss_derivatives = 2*(output_layer-ground_truth)
        last_layer_delta = loss_derivatives
        for layer in reversed(self.layers):
           last_layer_delta = layer.back_prop(last_layer_delta,self.learning_R)
        
    def save(self):
        for layer in self.layers:
            layer.save()
    
    def load(self):
        for layer in self.layers:
            layer.load()

    def train(self, input_layers, ground_truth, number_of_epochs):
        for epochs in range(number_of_epochs):
            self.feed_forward(input_layers)
            self.back_prop(ground_truth, self.output_layer)
            if epochs%100 == 0: 
                print(self.loss(ground_truth, self.output_layer))
        
        self.save()
        

