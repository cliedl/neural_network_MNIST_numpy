import numpy as np
import pandas as pd

from src.model import NN
from src.data import data_loader, process_data

if __name__ == "__main__":
    # Load data
    X_train, y_train, X_test, y_test = data_loader()

    # Process data
    X_train, y_train, X_test, y_test, y_train_onehot = process_data(X_train, y_train, X_test, y_test)

    # Create instance of neural network
    model = NN()

    # Training parameters
    learning_rate = 0.1
    epochs = 10
    batch_size = 64

    test_accuracy_list = []
    train_accuracy_list = []

    # Training loop
    for epoch in range(epochs):
        for i in range(0, len(X_train), batch_size):
            x_batch = X_train[i:i + batch_size]
            y_batch = y_train_onehot[i:i + batch_size]

            output = model.forward_pass(x_batch, y_batch)

            model.update_weights(learning_rate)

        # Train accuracy
        output = model.forward_pass(X_train, y_train, gradient=False)
        train_predictions = np.argmax(output, axis=1)
        train_accuracy = np.mean(train_predictions == y_train)
        train_accuracy_list.append(train_accuracy)

        # Test accuracy
        output = model.forward_pass(X_test, y_test, gradient=False)
        test_predictions = np.argmax(output, axis=1)
        test_accuracy = np.mean(test_predictions == y_test)
        test_accuracy_list.append(test_accuracy)

        print(f"Epoch {epoch + 1: >2}/{epochs}, Accuracy training set: {train_accuracy:.4f}, Accuracy test set: {test_accuracy:.4f}")


    results = { 'learning_rate': learning_rate,
                'epochs':epochs,
                'batch_size': batch_size,
                'test_accuracy': test_accuracy_list,
                'train_accuracy': train_accuracy_list}

    np.save('results_per_epoch.npy', results)
