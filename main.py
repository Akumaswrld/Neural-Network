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

def main():    
    # initialises the 2 layers in our network 4 being the number of data points in each batch and 5 being the number of neurons in the layer
    layer_1  = Layer(4,5)
    layer_2  = Layer(5,6)

    layer_1.forward(inputs = X)

    layer_2.forward(inputs = layer_1.get_output())

    print(layer_1.get_output())
    print(layer_2.get_output())

if __name__ == '__main__':
    main()



