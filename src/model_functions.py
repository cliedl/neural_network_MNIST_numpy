import numpy as np

def sigmoid(x, derivative=False):
    """Compute the sigmoid activation function.

    The sigmoid activation function is commonly used in machine learning and neural networks
    to map real-valued numbers to values between 0 and 1.

    Parameters:
    - x (float, array-like): The input value(s) to apply the sigmoid function to.

    Returns:
    - value (np.ndarray): It returns the sigmoid of the input 'x'.
    """
    value = 1 / (1 + np.exp(-x))
    return value

def softmax(x):
    """
    Compute the softmax activation function.

    The softmax activation function is commonly used in machine learning and neural networks
    to convert a vector of real numbers into a probability distribution over multiple classes.
    It exponentiates each element of the input vector and normalizes it to obtain the probabilities.

    Parameters:
    - x (numpy.ndarray): The input vector to apply the softmax function to.

    Returns:
    - value (np.ndarray): It returns the softmax of the input 'x', which is a probability distribution.]
    """

    # Numerically stable with large exponentials
    y = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return y / np.sum(y, axis=-1, keepdims=True)


def one_hot_enc(y, num_labels=10):
    """
    Convert class labels to one-hot encoded vectors.

    This function takes an array of class labels and converts them into one-hot encoded
    vectors. Each one-hot encoded vector represents the presence of a class label using a
    1.0 in the corresponding position and 0.0 elsewhere.

    Parameters:
    - y (array-like): An array of class labels to be one-hot encoded.
    - num_labels (int, optional): The total number of unique class labels. Defaults to 10.

    Returns:
    - one_hot (numpy.ndarray): A 2D numpy array where each column is a one-hot encoded
      vector representing a class label.
    """

    # Option 1
    # one_hot = np.zeros((len(y), num_labels), dtype=np.float32)
    # for i, x in enumerate(y):
    #   one_hot[i, x] = 1.

    # Option 2, more efficient
    one_hot = np.eye(num_labels)[y]

    return one_hot


