import numpy as np 
from random import randint

def main() -> None:    
    # Initialises a random number generator around the parameter number (0) meaning that in future, we can get a random number that is distributed in a normal distribution centred at 0
    np.random.seed(0)

    # Capitalised x denotes the training data 
    X = [[1.0, 2.0, 3.0, 2.5],
        [2.0, 5.0 ,-1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8]]
    
    layers = [
        Layer(4,10),
        Layer(10,10)
    ]

    activations = [
        ReLUActivation(),
        SoftmaxActivation()
    ]
    
    layer_output = Propagation.forward_propagate(input_data=X, layers=layers,activations=activations)
        
    # this will calculate the loss of the data batch given to the network 
    batch_loss_calc = Loss()
    batch_loss_calc.calculate(output=layer_output, intended_output=generate_one_hot_matrix(3,10))
    batch_loss = batch_loss_calc.get_loss()
    batch_accuracy = batch_loss_calc.get_accuracy()
    print('loss: ', batch_loss)
    print('Accuracy: ', batch_accuracy)

    weights = [layer.get_weights() for layer in layers]
    derivatives = [np.zeros_like(weight) for weight in weights]


class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.__weights = 0.1 * np.random.randn(n_inputs ,n_neurons)
        self.__biases = np.zeros((1, n_neurons))
        
    def forward(self, inputs):
        self.__output = np.dot(inputs, self.__weights) + self.__biases
    
    def get_output(self):
        return self.__output
    
    def get_weights(self):
        return self.__weights


class ReLUActivation:
    def forward(self, layer_outputs):
        self.__output = np.maximum(0, layer_outputs)
    
    def get_output(self):
        return self.__output 


class SoftmaxActivation:
    def forward(self, layer_outputs):
        exp_values = np.exp(layer_outputs - np.max(layer_outputs, axis=1, keepdims=True)) 
        self.__output = exp_values / np.sum(exp_values, axis=1, keepdims=True )

    
    def get_output(self):
        return self.__output


class Loss:
    def calculate(self, output, intended_output):
        sample_losses = self.forward(pred=output, true_values=intended_output)
        self.__data_loss = np.mean(sample_losses)
        

    def forward(self, pred, true_values):
        samples = len(pred)
        pred_clipped_values = np.clip(pred, 1e-7, 1-(1e-7))

        if len(true_values.shape) == 1:
            correct_values = pred_clipped_values[range(samples), true_values]
        else:
            correct_values = np.sum(pred_clipped_values * true_values, axis=1)
        
        self.calculate_accuracy(pred=pred, true_values=true_values)
        return -np.log(correct_values)
    
    def calculate_accuracy(self, pred, true_values):
        if len(true_values) == 1:
            self.__accuracy = np.mean(np.argmax(pred, axis=1) == true_values)
        else:
            self.__accuracy = np.mean(np.argmax(pred, axis=1) == np.argmax(true_values, axis=1))
        
        
    def get_loss(self):
        return self.__data_loss
    
    
    def get_accuracy(self):
        return self.__accuracy
    


class Propagation:
    def forward_propagate(layers, activations, input_data):
        for i in range(len(layers)):
            layers[i].forward(inputs=input_data)
            layer_output = layers[i].get_output()
            activations[i].forward(layer_outputs = layer_output)
            input_data = activations[i].get_output()
        
        return input_data
    def backward_propagate():
        pass


# for testing purposes

def generate_one_hot_matrix(length, num_classes):
    matrix = np.zeros((length, num_classes), dtype=int)
    for i in range(length):
        matrix[i, randint(0, num_classes-1)] = 1
    # print(matrix)
    return matrix


if __name__ == '__main__':
     main()




