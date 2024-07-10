import numpy as np 

# Initialises a random number generator around the parameter number (0) meaning that in future, we can get a random number that is distributed in a normal distribution centred at 0
np.random.seed(0)

# Capitalised x denotes the training data 
X = [[1.0, 2.0, 3.0, 2.5],
     [2.0, 5.0 ,-1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.__weights = 0.1 * np.random.randn(n_inputs ,n_neurons)
        self.__biases = np.zeros((1, n_neurons))
        
    def forward(self, inputs):
        self.__output = np.dot(inputs, self.__weights) + self.__biases
    
    def get_output(self):
        return self.__output


class ActivationFunction:
    def forward(self, inputs):
        self.__output = np.maximum(0, inputs)
    
    def get_forward_output(self):
        return self.__output 


class SoftmaxActivation:
    def forward(self, layer_outputs):
        exp_values = np.exp(layer_outputs - np.max(layer_outputs, axis=1, keepdims=True)) 
        self.__output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def get_forward_output(self):
        return self.__output


def main():    
    pass

if __name__ == '__main__':
    main()



