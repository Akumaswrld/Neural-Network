import numpy as np 
from random import randint

def main() -> None:    
    # Initialises a random number generator around the parameter number (0) meaning that in future, we can get a random number that is distributed in a normal distribution centred at 0
    np.random.seed(0)

    # Capitalised x denotes the training data 
    X = [[1.0, 2.0, 3.0, 2.5],
        [2.0, 5.0 ,-1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8]]
    
    # initialising first Hidden Layer and its activation function 
    layer_1 = Layer(4,5)
    activation_1 = ActivationFunction()

    # apply inputs*weights + biases -> Activation function (Relu in this case)
    layer_1.forward(inputs=X)
    layer_1_output = layer_1.get_output()
    activation_1.forward(inputs=layer_1_output)
    activation_1_output = activation_1.get_forward_output()

    # initialising second Hidden layer and softmax activation function 
    layer_2 = Layer(5,5)
    activation_2 = SoftmaxActivation()

    # just like before except this time we using softmax function -> e^(single output) / sum of all outputs raised to power of e -> giving a probability distribution
    layer_2.forward(activation_1_output)
    layer_2_output = layer_2.get_output()
    activation_2.forward(layer_outputs=layer_2_output)
    activation_2_output = activation_2.get_forward_output()

    # this will calculate the loss of the data batch given to the network 
    batch_loss_calc = Loss()
    batch_loss_calc.calculate(output=activation_2_output, intended_output=generate_one_hot_matrix(3,5))
    batch_loss = batch_loss_calc.get_loss()
    batch_accuracy = batch_loss_calc.get_accuracy()
    print('loss: ', batch_loss)
    print('Accuracy: ', batch_accuracy)
    

    
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
        self.__output = exp_values / np.sum(exp_values, axis=1, keepdims=True )
        

    
    def get_forward_output(self):
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
    

# for testing purposes

def generate_one_hot_matrix(length, num_classes):
    matrix = np.zeros((length, num_classes), dtype=int)
    for i in range(length):
        matrix[i, randint(0, num_classes-1)] = 1
    return matrix

if __name__ == '__main__':
     main()




