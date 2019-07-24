import numpy as np
import math

"""
 ==================================
 Problem 3: Neural Network Basics
 ==================================
    Generates a neural network with the following architecture:
        Fully connected neural network.
        Input vector takes in two features.
        One hidden layer with three neurons whose activation function is ReLU.
        One output neuron whose activation function is the identity function.
"""


#pragma: coderesponse template
def rectified_linear_unit(x):
    """ Returns the ReLU of x, or the maximum between 0 and x."""
    # TODO
    return np.maximum(x,0)

#pragma: coderesponse end

#pragma: coderesponse template
def rectified_linear_unit_derivative(x):
    """ Returns the derivative of ReLU."""
    # TODO
    if x>0:
        return 1
    else:
        return 0
#pragma: coderesponse end

def output_layer_activation(x):
    """ Linear function, returns input as is. """
    return x

def output_layer_activation_derivative(x):
    """ Returns the derivative of a linear function: 1. """
    return 1

class NeuralNetwork():
    """
        Contains the following functions:
            -train: tunes parameters of the neural network based on error obtained from forward propagation.
            -predict: predicts the label of a feature vector based on the class's parameters.
            -train_neural_network: trains a neural network over all the data points for the specified number of epochs during initialization of the class.
            -test_neural_network: uses the parameters specified at the time in order to test that the neural network classifies the points given in testing_points within a margin of error.
    """

    def __init__(self):

        # DO NOT CHANGE PARAMETERS
        self.input_to_hidden_weights = np.matrix('1 1; 1 1; 1 1')
        self.hidden_to_output_weights = np.matrix('1 1 1')
        self.biases = np.matrix('0; 0; 0')
        self.learning_rate = .001
        self.epochs_to_train = 10
        self.training_points = [((2,1), 10), ((3,3), 21), ((4,5), 32), ((6, 6), 42)]
        self.testing_points = [(1,1), (2,2), (3,3), (5,5), (10,10)]

#pragma: coderesponse template prefix="class NeuralNetwork(NeuralNetworkBase):\n\n"
    def train(self, x1, x2, y):
     
        ### Remapping variable names
        W1 = np.array(self.input_to_hidden_weights)
        b = np.array(self.biases)
        W2 = np.array(self.hidden_to_output_weights)
        f1 = np.vectorize(rectified_linear_unit)
        df1 = np.vectorize(rectified_linear_unit_derivative)
        f2 = output_layer_activation
        df2 = output_layer_activation_derivative
        ### Forward propagation ###
        # input_values = np.matrix([[x1],[x2]]) # 2 by 1
        x = np.array([[x1], [x2]])
    

        # Calculate the input and activation of the hidden layer
        #hidden_layer_weighted_input = # TODO (3 by 1 matrix)
        #z1 = np.matmul(W1,x) + b
        z1 = W1 @ x + b

        #hidden_layer_activation = # TODO (3 by 1 matrix)
        a1 = f1(z1)

        #output =  # TODO
        #u1 = np.matmul(W2,a1)
        u1 = W2 @ a1
        
        #activated_output = # TODO
        o1 = f2(u1)

        ### Backpropagation ###

        # Compute gradients
        #output_layer_error = o1 - y
        #hidden_layer_error = # TODO (3 by 1 matrix)
        #W2.shape = 3,1

        #bias_gradients = np.multiply(np.multiply((o1 - y),df1(z1)),W2.T)
        #hidden_to_output_weight_gradients = np.multiply((o1 - y),a1)
        #input_to_hidden_weight_gradients = np.matmul(np.multiply(np.multiply((o1 - y),df1(z1)),W2.T), x.T)
        
        bias_gradients = (df1(z1) * W2.T) @ (o1 - y)
        hidden_to_output_weight_gradients = a1 @ (o1 - y)
        input_to_hidden_weight_gradients = (df1(z1) * W2.T) @ (o1 - y) @ x.T
        

        # Use gradients to adjust weights and biases using gradient descent
        self.biases = b - self.learning_rate * bias_gradients
        self.input_to_hidden_weights = W1 - self.learning_rate * input_to_hidden_weight_gradients
        self.hidden_to_output_weights = W2 - self.learning_rate * hidden_to_output_weight_gradients.T


        

#pragma: coderesponse end

#pragma: coderesponse template prefix="class NeuralNetwork(NeuralNetworkBase):\n\n"
    
    def predict(self, x1, x2):
        
        # Compute output for a single input(should be same as the forward propagation in training)
        #hidden_layer_weighted_input = # TODO
        #hidden_layer_activation = # TODO
        #output = # TODO
        #activated_output = # TODO
        
        ### Remapping variable names
        W1 = np.array(self.input_to_hidden_weights)
        b = np.array(self.biases)
        W2 = np.array(self.hidden_to_output_weights)
        f1 = np.vectorize(rectified_linear_unit)
        df1 = np.vectorize(rectified_linear_unit_derivative)
        f2 = output_layer_activation
        df2 = output_layer_activation_derivative
        ### Forward propagation ###
        # input_values = np.matrix([[x1],[x2]]) # 2 by 1
        x = np.array([[x1], [x2]])
        
        #convert to numpy array
        #W1 = np.array(W1)
        #b = np.array(b)
        #W2 = np.array(W2)
        #W2.shape = (1,3)

        # Calculate the input and activation of the hidden layer
        #hidden_layer_weighted_input = # TODO (3 by 1 matrix)
        z1 = W1 @ x + b

        #hidden_layer_activation = # TODO (3 by 1 matrix)
        a1 = f1(z1)

        #output =  # TODO
        #u1 = np.matmul(W2,a1)
        u1 = W2 @ a1
        
        #activated_output = # TODO
        o1 = f2(u1)
        activated_output = o1

        return activated_output.item()
#pragma: coderesponse end
        

    # Run this to train your neural network once you complete the train method
    def train_neural_network(self):

        for epoch in range(self.epochs_to_train):
            for x,y in self.training_points:
                self.train(x[0], x[1], y)

    # Run this to test your neural network implementation for correctness after it is trained
    def test_neural_network(self):

        for point in self.testing_points:
            print("Point,", point, "Prediction,", self.predict(point[0], point[1]))
            if abs(self.predict(point[0], point[1]) - 7*point[0]) < 0.1:
                print("Test Passed")
            else:
                print("Point ", point[0], point[1], " failed to be predicted correctly.")
                return

x = NeuralNetwork()

x.train_neural_network()

# UNCOMMENT THE LINE BELOW TO TEST YOUR NEURAL NETWORK
x.test_neural_network()
