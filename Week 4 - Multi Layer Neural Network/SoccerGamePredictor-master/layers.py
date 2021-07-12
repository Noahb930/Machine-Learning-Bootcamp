import numpy as np
import os
import itertools

class Layer:
    counter = itertools.count()
    def activation(self,func,deriv,x):
        
        if not deriv :
            if func == self.func:
                return 1 / (1 + np.exp(-x))
            elif func == "relu":
                if x > 0:
                    return x
                else:
    	            return 0
        else:
            if func == self.func:
              return x * (1.0 - x)
            elif func == "relu":
                if x>0:
                    return 1
                else:
                    return 0
    def save(self):
        np.savez(os.getcwd()+ "/layers/layer"+str(self.id)+".npz",weights =self.weights, bias=self.bias)
    def load(self):
        self.weights =  np.load(os.getcwd()+ "/layers/layer"+str(self.id)+".npz")["weights"]
        self.bias =  np.load(os.getcwd()+ "/layers/layer"+str(self.id)+".npz")["bias"]
class FullyConnectedLayer(Layer):
     def __init__(self, func, hidden_size, output_size):
        self.id = next(self.counter)
        self.func = func
        self.weights = np.random.randn(hidden_size, output_size)  
        self.bias = np.random.randn(1,output_size) 

     def feed_forward(self,input_layer):
        self.input_layer = input_layer
        self.z = np.dot(input_layer,self.weights) + self.bias
        self.output = self.activation(self.func, False, self.z)
        return self.output
    
     def back_prop(self,prev_layer_delta, lr):
        self.d_z = prev_layer_delta*(self.activation(self.func,True, self.output))
        self.d_w = self.input_layer.T.dot(self.d_z)        
       
        self.d_x = self.d_z.dot(self.weights.T)
        self.d_b = np.sum(self.d_z,axis=0)
        
        self.bias = self.bias - (self.d_b*lr)
        self.weights = self.weights - (self.d_w*lr)
        return self.d_x


        








