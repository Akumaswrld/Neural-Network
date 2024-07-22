# Handwritten Digit Classification Model
This is my short documentation on my code (ignore misspellings or typos, i wrote this at 2am GMT)
 
 # INTRODUCTION:
 This neural network is a multi class classification model that can classify digits 0-9 using the mnist large data set. It took me around 2 months to learn the theory needed to build the model and actually implementing it took around 2-3 weeks. In this document i will be explaining all the functions/classes and a bit of the theory behind them (obviously not in crazy detail, im saving that for my real NEA)

 ***MODULES USED***:

 the modules i used are minimal, no large machine learning library was used as that beats the whole purpose of why im doing this. All i used was:
 - **Numpy:** this module was used to do all the complex matrix operations for us most optimally 
 - **Pandas:** i used pandas to import parts of the data from the large data set without having to import the whole dataset into my memory every time i trained the model
 - **PIL:** i only used PIL right at the end to test the model using my own hand written digits
 this failed horribly because i had to reformat the image so its 28x28, at first i thought this would be fine but then my model that had 95% accuracy on testing data, was performing at less than 50% on my own images, i then used PIL to show me the image and oh god, i couldnt even tell what the digit was myself. So i was limited to using testing data provided by the data set


 # NEURAL NETWORK CLASS:
 this is the main neural network class as you can probably tell, when an instance of the class is made, this is the actual model, it will initialise all the hidden layers using Layers objects (will explain more about this when i talk about the layers class)
    
***forward_propagate method:***

this is the first actual function performed by the neural network, the input data is passed through the network from the input layer through the hidden layers and an output is produced by the output layer.

quick run through of how this works:
every node from a layer is connected to every node in the next layer by what is known as a weight (a scalar quantity which is multiplied to the output of the node which is the input of the next node), a bias associated with the next node is added along at the end.

so basically every time an output travels to the next layer it goes through a sequence of operation which we can describe as one function:

*f(x) = wx + b* --> where x is the ouput of the previous layer (input for the current layer) w is the weight and b is the bias 

once this has been computed, for the current layer, the output of this function which is now passed into a node has to be passed through a function associated with that layer known as an activation function, this activation function introduces non-linearity to the data which makes it easier for the network to predict more complicated data. 
examples of activation functions are:
- *ReLU (Rectified linear unit)*
- *Sigmoid (1/1+e^(-x))* 
- *Hyperbolic Tan (tanh)*
- *Softmax* 

for my model, i decided to go for ReLU for the activation of the first hidden layer and softmax for the activation of the second hidden layer (will go into more detail when i reach their individual functions)

once this is all computed the softmax activation returns a probability distribution of its certainty of what the outcome should be.

***backward_propagate method:***

the mechanics of how backward propagation works is that it calculates the loss of the model (how far it was off the actual output), then it calculates the derivative of the loss function with respect to every weight in each layer and find where the derivative is the most negative (slopes downwards the most) as this is where the the loss function would be at its lowest so in turn the loss of the model will be at a minimum. The gradient is then multiplied by a hyperparemeter known as the learning rate and this is taken away from the corresponding weight in the corresponding layer.

wrapping my head around the concept of backward propagation and how the partial derivatives of the loss function worked was extremely difficult, it took me 2 whole weeks to finally understand how it worked and how to implement it.

in all the lectures and tutorials i watched, they used a loop to calculate all the derivates however, this was a bit too much for me to understand so, since i only had two sets of weights (between input and hidden layer 1, between hidden layer 1 and hidden layer 2), i decided to manually calculate the derivatives as, this way , i actually understood what i was doing because i was manually propagating the loss backward throughout the layers.

using the chain rule, to find the derivative of the loss function with respect to a weight at the second hidden layer, i'd need to multiply the derivative of the activation function of hidden layer 2 (this turned out to be the final output from the output layer - the target values in one hot encoded format) with the inputs of layer 2 and all divided by the batch size.

This is where propagating the loss through the layers comes in, to find the derivative of the loss function with respect to the weights from the first hidden layer i'd need:
- the derivative of the activation function from layer 2 (what we used in the previous calculations, this is what is meant by propagating the loss backward) multiplied by the weights at layer 2 
- the derivative of the activation function of layer 1 using the original provided data 

these two are then multiplied with each other and divided by the batch size

***update_params method:***

this method just takes the derivatives from back prop, multiplies them by a hyper parameter (learning rate) and takes this away from the corresponding weights from the corresponding layer 

***train method:***

takes 4 parameters: learning_rate, epoches, input_data and targets
- learning rate: a hyperparameter that dictates how far the weights are tweaked after each iteration, it is important for this to be optimal for the model so that it can converge at a local minima (if set too high, it could potentially diverge and if too low, it may never converge)
-epoches: the number of iterations through the same training data 
-input_data: the data? how else do i say this
-targets: the correct ouput for the corresponding data

the train method applies forward and back propagation to the same data for many iterations and the weights and biases are tweaked each time 

after every 10 iterations, the loss and accuracy of the model in that iteration are printed into terminal just to give an update on the training of the model 

***predict method:***

once training is finished, you can use the predict method to test the accuracy of the model on data it has never seen before to ensure that no overfitting has occurred, this method outputs the prediction along with the probability 

***save_model method:***

after training and testing is done, if the weights and biases produce an accurate result in the testing phase, you can save the weights and biases into a .npz file (a file format to store numpy arrays)

***load_model method:***

simply loads a model from a .npz file 
 
# LAYERS CLASS:
this is a class of layers, initialised with 2 dimensions, the number of rows being the number of inputs it will receive and columns being the number of nodes in that layer (this is to ensure that the dimensions are correct to perform matrix multiplication with the input matrix). With these dimensions, a set of weights are randomly initialised and with the same dimensions, a matrix of 0's is created as the initial biases

***forward method:***

this simply performs the input * weight + bias i spoke about in forward propagation 



# FUNCTIONS CLASS:
this is a class of static methods that i just for general organisation, this includes all the functions that are required for forward and back prop to work that are did not directly need to be related to the neural network class for example, one_hot_encoding, this is just a general method. However i am still uncertain whether i shouldve included the activation functions here, but it still worked out fine so i guess theyll stay there

***relu method:***

this takes a numpy array as an input and applies the relu function to every value inside the array. If the value if less than or equal to 0, it is set to a 0 else it is left unchanged.

***relu_derivative method:***

this takes a numpy array and calculates the gradient of each value in the array, if the value is 0, the gradient is 0 or less, the derivative will be set to a 0 else it would be set to a 1.

***softmax method:***

this one is very important:
the issue: what i want to do is form a probability distribution based on the data provided, however we may have some negative values flying around so we cannot simply sum these values together because it will result in an inaccurate representation of all the values. At the same time, we cannot simply take the modulus of the negative values because now a -5 will have the same impact as a +5 which will cause many problems. So instead, we take e^(value) because (as e^x > 0 for all real values of x) we will only have positive values without the previously negative values losing the significance of their negativity. However, quickly we run into another problem, what if the value is relatively big? This will cause e^(value) to quickly shoot off to a large value which may cause problems with our neural network or may even result in the number being too big for the program to store it. So before we exponentiate each value, we take away from it the largest number in the set, therefore we will only have numbers less than 0 and will stop the exponential function from creating absurdly large values. This will not effect our final outcome as we are taking away a set quantity from each value (not multiplying or dividing) so its like shifting every number down in the number line by x amount, by doing all this, the exponential function will only give is values between 0 and 1 which will not cause any problems for our model. Now we do what we'd do for any probability distribution, we divide each value by the sum of all the values in that set.

***categorical_cross_entropy method:***

since we know that the predictions parameter will always be an matrix of numbers between 0-1 (softmax activation), we get the probability of the actual output we want and take the - of the natural log of that value. However, what if the value is 0? recall that the domain of the Ln function is x > 0, so if we enter 0 into the natural log function, it would break. This is where we create an interval where all the values should fit this being 1 x 10^-7 and 1-(1 x 10^-7), this means every value will now be between these 2 numbers thus the log function will never be getting a 0 inputted. The value at the end is the loss of the model in that iteration which is outputted in the terminal when training the model. Other than that i dont think its ever used, i was initially going to use this for loss in back propagation but i realised that what i was doing was incorrect.

***accuracy method:***

return a % of how many predictions the model got in a batch 

***one_hot method:***

takes a 1 dimensional vector and one hot encodes it for example:
array = [1,3,4]
one_hot_array = [[0,1,0,0,0],
                    [0,0,0,1,0],
                    [0,0,0,0,1]]
each value in the vector is an 'index' of the correct output, one hot creates a vector for each value where there is a 1 in the index given by the value

***load_image method:***

used PIL library to import and use my own pictures (this did not work out well), i did this by getting the image and then reformatting it to 28x28 which is the format given by the dataset and the format that my model has been trained on. I then turn this into a numpy array where each value represents a pixel. I then flattened this matrix into a vector which is the format which is given by the data set.


# Issues:
one issue i had with the model was when i had 'finished' the whole model bearing in mind this was my 4th attempt of making this model. When i first tried to train the model i had an issue where i could see the model converging to a minimum loss however after 100 iterations it would oscillate between 2 loss values forever and the model was never learning. After hours of researching, i realised i had to normalise the data for my model to be efficient with it. Basically in the data set, each pixel has a value 0-255, 0 being white and 255 being black and shades of gray in between, so i had to divide all of the values by 255 so that i will only have values between 0 and 1. As soon as i made a change, i saw a convergence of 89% accuracy after 50 iterations. 

After 2 days of testing and tweaking hyper parameters, I have managed to train my model so that it can classify unseen testing data with 95% accuracy

    
    
    

         

    
