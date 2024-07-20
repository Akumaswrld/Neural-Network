import numpy as np
import pandas as pd

class NeuralNetwork:
    def __init__(self):
        self.__layer_1 = Layers(784, 10)
        self.__layer_2 = Layers(10,10)

    def forward_propagate(self, inputs):
        self.layer_1_pre_activation = self.__layer_1.forward(inputs=inputs)
        self.layer_1_activation = Functions.relu(pre_activations=self.layer_1_pre_activation)

        self.layer_2_pre_activation = self.__layer_2.forward(inputs=self.layer_1_activation)
        self.layer_2_activation = Functions.softmax(pre_activations=self.layer_2_pre_activation)

        self.__final_output = self.layer_2_activation

        return self.__final_output

    def backward_propagate(self, inputs, targets): 
        m = targets.size
        targets_one_hot = Functions.one_hot(targets=targets)
        delta_l2 = self.__final_output - targets_one_hot
        g_W2 = (1/m) * np.dot(self.layer_1_activation.T, delta_l2)
        g_B2 = (1/m) * np.sum(delta_l2, axis=0, keepdims=True)
        delta_l1 = np.dot(delta_l2, self.__layer_2.weights.T) * Functions.relu_derivative(self.layer_1_pre_activation)
        g_W1 = (1/m) * np.dot(inputs.T, delta_l1)
        g_B1 = (1/m) * np.sum(delta_l1, axis=0, keepdims=True)

        return g_W1, g_B1, g_W2, g_B2


    def update_params(self, g_W1, g_B1, g_W2, g_B2, learning_rate): # also can be said that this is gradient descent
        self.__layer_1.weights -= learning_rate * g_W1
        self.__layer_1.biases -= learning_rate * g_B1
        self.__layer_2.weights -= learning_rate * g_W2
        self.__layer_2.biases -= learning_rate * g_B2 
        

    def train(self, learning_rate, epoches, input_data, targets): # this is the method we use to train our model 
        for n in range(epoches):
            output = self.forward_propagate(inputs=input_data)
            g_W1, g_B1, g_W2, g_B2 = self.backward_propagate(inputs= input_data, targets=targets)
            self.update_params(g_W1, g_B1, g_W2, g_B2, learning_rate)

            if n % 10 == 0:
                loss = Functions.categorical_cross_entropy(predictions=output, targets=targets)
                accuracy = Functions.accuracy(predictions=output, targets=targets)
                print(f'loss in epoche {n}: ', loss)
                print(f'accuracy: ', accuracy)

    def predict(self, data): # this is the method after the network has been trained so that it can predict 
        prediction = self.forward_propagate(inputs = data)
        index = np.argmax(prediction)
        value = prediction[0][index] * 100

        print(f'I am {value}% certain that this digit is a {index}')

class Layers:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2.0 / n_inputs)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        return np.dot(inputs, self.weights) + self.biases


class Functions:
    def relu(pre_activations):
        return np.maximum(0, pre_activations)

    def relu_derivative(matrix):
        return np.where(matrix > 0, 1, 0)
    
    def softmax(pre_activations):
        exp = np.exp(pre_activations - np.max(pre_activations, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)
    
    def categorical_cross_entropy(predictions, targets):
        samples = len(predictions)
        pred_clipped_values = np.clip(predictions, 1e-7, 1-(1e-7))
        correct_values = pred_clipped_values[range(samples), targets]
        log_values = -np.log(correct_values)

        return np.mean(log_values)
    
    def accuracy(predictions, targets):
        return np.mean(np.argmax(predictions, axis=1) == targets)
    
    def one_hot(targets):
        one_hot_targets = np.zeros((targets.size, 10))
        one_hot_targets[np.arange(targets.size), targets] = 1
        return one_hot_targets


def main() -> None:
    data = np.array(pd.read_csv('mnist_train.csv', nrows=1000))
    test = np.array(pd.read_csv('mnist_test.csv', nrows=20))
    np.random.shuffle(data)

    
    targets = data[:, 0] 
    data = data[:, 1:] / 255

    test_targets = test[:, 0]
    test = test[:, 1:] / 255


    neural_network = NeuralNetwork()
    neural_network.train(learning_rate=0.15, epoches=100, input_data=data, targets=targets)

    for data in test:
        neural_network.predict(data=data)
        
    print(test_targets)
    

    

    # (self, learning_rate, epoches, input_data, targets)



if __name__ == '__main__':
    main()

