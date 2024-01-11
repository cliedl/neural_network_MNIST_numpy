import numpy as np
from .model_functions import sigmoid, softmax

class NN:
    """
    Simple neural network with three layers, implemented with numpy
    """
    def __init__(self,
                    input_size=784,
                    hidden1_size=128,
                    hidden2_size=64,
                    output_size=10):

        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size

        # Initialize weights
        self.W1, self.W2, self.W3, self.b1, self.b2, self.b3 = self.initialize_wb()

        np.random.seed(42)

    def initialize_wb(self):
        """
        Initialize weights and biases for a neural network.

        Args:
        - input_size (int): Number of input features.
        - hidden1_size (int): Number of neurons in the first hidden layer.
        - hidden2_size (int): Number of neurons in the second hidden layer.
        - output_size (int): Number of output neurons (number of classes).

        Returns:
        - W1, W2, W3 (np.ndarray): 2D arrays for the weights with random values from a standard normal
            distribution
        - b1, b2, b3: (np.ndarray): 2D arrays for the bias as zeros
        # your code here
        """

        # Initialize weights with random values from a standard normal distribution
        W1 = np.random.randn(self.input_size, self.hidden1_size)  # size input x hidden1
        W2 = np.random.randn(self.hidden1_size, self.hidden2_size)  # size hidden1 x hidden2
        W3 = np.random.randn(self.hidden2_size, self.output_size)  # size hidden2 x output

        # Initialize biases as zeros
        b1 = np.zeros((1, self.hidden1_size))  # size 1 x hidden1
        b2 = np.zeros((1, self.hidden2_size))  # size 1 x hidden2
        b3 = np.zeros((1, self.output_size))  # size 1 x output

        return W1, W2, W3, b1, b2, b3

    def forward_pass(self, x_batch, y_batch, gradient=True):
        """
        Args:
            x_batch (numpy.ndarray): input vector
            y_batch (numpy.ndarray): target vector
            gradient (Boolean): Whether gradient should be calculated or not

        Returns:
            output (nump.ndarray): output vector
        """

        # Forward pass
        z1 = np.dot(x_batch, self.W1) + self.b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = sigmoid(z2)
        z3 = np.dot(a2, self.W3) + self.b3
        output = softmax(z3)

        if gradient:
            # Backpropagation, update derivatives
            delta = output - y_batch
            self.dW3 = np.dot(a2.T, delta)
            self.db3 = np.sum(delta, axis=0, keepdims=True)
            delta = np.dot(delta, self.W3.T) * (a2 * (1 - a2))
            self.dW2 = np.dot(a1.T, delta)
            self.db2 = np.sum(delta, axis=0, keepdims=True)
            delta = np.dot(delta, self.W2.T) * (a1 * (1 - a1))
            self.dW1 = np.dot(x_batch.T, delta)
            self.db1 = np.sum(delta, axis=0, keepdims=True)

        return output

    def update_weights(self, learning_rate=0.1):
        """
        Updates weight by taking a step of size learning_rate into
        the direction of the negative gradient

        Args:
            learning_rate (float): learning rate

        Returns: -
        """
        self.W1 -= learning_rate * self.dW1
        self.b1 -= learning_rate * self.db1
        self.W2 -= learning_rate * self.dW2
        self.b2 -= learning_rate * self.db2
        self.W3 -= learning_rate * self.dW3
        self.b3 -= learning_rate * self.db3

        return







