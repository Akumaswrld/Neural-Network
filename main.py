import numpy as np 

def main():    
    # Initialises a random number generator around the parameter number (0) meaning that in future, we can get a random number that is distributed in a normal distribution centred at 0
    np.random.seed(0)

    # Capitalised x denotes the training data 
    X = [[1.0, 2.0, 3.0, 2.5],
        [2.0, 5.0 ,-1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8]]
    
    layer_1 = Layer(n_inputs=4, n_neurons=3)
    layer_1.forward(inputs=X)
    layer_1_outputs = layer_1.get_output()
    # print(layer_1_outputs)

    layer_1_activation = ActivationFunction()
    layer_1_activation.forward(layer_1_outputs)
    
    layer_1_softmax = SoftmaxActivation()
    layer_1_softmax.forward(layer_outputs=layer_1_activation.get_forward_output())


    
class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.__weights = 0.1 * np.random.randn(n_inputs ,n_neurons)
        # print(self.__weights)
        self.__biases = np.zeros((1, n_neurons))
        # print(self.__biases)
        
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
        self.__output = exp_values / np.sum(exp_values, axis=1, keepdims=True )

        print(self.__output)
        # print(np.sum(self.__output, keepdims=True, axis=1))

    
    def get_forward_output(self):
        return self.__output


class Loss:
    def calculate(self, output, intended_output):
        sample_losses = self.forward(pred=output, true=intended_output)
        data_loss = np.mean(sample_losses)

    def forward(self, pred, true_values):
        samples = len(pred)
        pred_clipped_values = np.clip(pred, 1e-7, 1-(1e-7))

        if len(true_values.chape) == 1:
            correct_values = pred_clipped_values[range(samples), true_values]
        else:
            correct_values = np.sum(pred_clipped_values * true_values, axis=1)



if __name__ == '__main__':
    main()



