import mnist
from .model_functions import one_hot_enc

def data_loader():
    """
    Load and return the MNIST dataset for training and testing.

    Returns:
    - X_train (numpy.ndarray): Training set, a 3D array of shape (num_samples, 28, 28),
      where num_samples is the number of training samples.
    - y_train (numpy.ndarray): Training labels, a 1D array of shape (num_samples,) containing
      the corresponding class labels for the training images.
    - X_test (numpy.ndarray): Test set, a 3D array of shape (num_samples, 28, 28),
      where num_samples is the number of test samples.
    - y_test (numpy.ndarray): Test labels, a 1D array of shape (num_samples,) containing
      the corresponding class labels for the test images.

    Example usage:
    >>> X_train, y_train, X_test, y_test = data_loader()
    >>> print("Training set shape:", X_train.shape)
    >>> print("Training labels shape:", y_train.shape)
    >>> print("Test set shape:", X_test.shape)
    >>> print("Test labels shape:", y_test.shape)
    """
    X_train = mnist.train_images()
    y_train = mnist.train_labels()
    
    X_test = mnist.test_images()
    y_test = mnist.test_labels()

    return X_train, y_train, X_test, y_test


def process_data(X_train, y_train, X_test, y_test):
    """
    Processes data: Normalizes features to range [0, 1] and flattens feature vector

    Args:
        X_train:
        y_train:
        X_test:
        y_test:

    X_train,
    Returns:
        X_train: normalized and flattened
        y_train: as input
        X_test: normalized and flattened
        y_test: as input
        y_train_onehot: one hot encoding of y_train
    """
    # Normalize to 1
    X_train = X_train / 255.
    X_test = X_test / 255.

    # Flatten feature vectors
    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))

    # One-hot encoding
    num_classes = 10
    # your code here
    y_train_onehot = one_hot_enc(y_train, num_classes)

    return X_train, y_train, X_test, y_test, y_train_onehot