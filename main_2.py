import numpy as np 
from random import randint

class NeuralNetwork:
    def __init__(self):
        self.__layers = [
            Layer(4,10),
            Layer(10,10)
        ]        
        self.__activation_functions = [
            ReLUActivation(),
            SoftmaxActivation()
        ]
        self.__weights = [layer.get_weights() for layer in self.__layers]
        self.__biases = [layer.get_biases() for layer in self.__layers]
        self.__learning_rate = 0.1

    def forward_propagate(self, input_data):
        self.__activations = []
        self.__pre_activations = []
        for i in range(len(self.__layers)):
            self.__layers[i].forward(inputs=input_data)
            layer_output = self.__layers[i].get_output()
            self.__pre_activations.append(layer_output)
            self.__activation_functions[i].forward(layer_outputs = layer_output)
            input_data = self.__activation_functions[i].get_output()
            self.__activations.append(input_data)

        self.__layer_outputs = input_data # this is just so that it makes sense that were returning outputs rather than 'inputs' even though the variables hold the same values
        self.__activations.append(self.__layer_outputs)
        self.__pre_activations.append(self.__layer_outputs)
        return self.__layer_outputs

    def backward_propagate(self, input_data, intended_output):
        # first calculate the totoal batch loss 
        loss_function = Loss()
        loss_function.calculate(output=self.__layer_outputs, intended_output=intended_output)
        error = loss_function.get_loss()
        delta_l2 = self.__layer_outputs - intended_output # loss gradient
        print('error: ', error)

        # calculate the derivative of error with respect to each weight
        output_layer_activations = self.__activations[1]
        g_W2 = (np.dot(output_layer_activations.T, delta_l2)) / len(output_layer_activations)
        g_B2 = np.sum(delta_l2, axis=0) / len(output_layer_activations)

        delta_l1 = np.dot(delta_l2, self.__weights[-1].T) * ReLUActivation.derivative(self.__pre_activations[0])
        g_W1 = (np.dot(input_data.T, delta_l1)) / len(output_layer_activations)
        g_B1 = (np.sum(delta_l1, axis = 0))/len(output_layer_activations)

        self.__derivatives = [g_W1, g_W2]
        self.__bias_derivatives = [g_B1, g_B2]

        
    def train(self, epochs, learning_rate):
        pass

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

    def get_biases(self):
        return self.__biases
    

class ReLUActivation:
    def forward(self, layer_outputs):
        self.__output = np.maximum(0, layer_outputs)
    
    def get_output(self):
        return self.__output 
    
    def derivative(matrix):
        return np.where(matrix > 0, 1, 0)



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
    

def generate_one_hot_matrix(length, num_classes):
    matrix = np.zeros((length, num_classes), dtype=int)
    for i in range(length):
        matrix[i, randint(0, num_classes-1)] = 1
    # print(matrix)
    return matrix

def main() -> None:    
    # Initialises a random number generator around the parameter number (0) meaning that in future, we can get a random number that is distributed in a normal distribution centred at 0
    np.random.seed(0)

    # Capitalised x denotes the training data 
    dummy_data = np.array([[1.0, 2.0, 3.0, 2.5],
                           [2.0, 5.0 ,-1.0, 2.0],
                           [-1.5, 2.7, 3.3, -0.8]])
    
    neural_network = NeuralNetwork()
    activations = neural_network.forward_propagate(input_data=dummy_data)
    print(activations)
    neural_network.backward_propagate(input_data=dummy_data, intended_output=generate_one_hot_matrix(3,10))
 

if __name__ == '__main__':
    main()