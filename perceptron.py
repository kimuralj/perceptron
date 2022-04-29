import numpy as np

class Perceptron:
    def __init__(self, eta=0.01, iteractions=1000, classes=2):
        self.eta = eta
        self.iteractions = iteractions
        self.number_of_classes = classes
        self.W = None

    # Defines the sigmoid function
    def sigmoid_func(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self, X, Y):
        n_samples, n_inputs = X.shape

        n_inputs = n_inputs + 1  # include 'bias input'
        # Initialize with random weights, bias is included
        if self.number_of_classes > 2:
            # Create W matrix with (n_inputs, self.number_of_classes) size
            self.W = np.random.random((n_inputs, self.number_of_classes))
        else:
            # Create W matrix with (n_inputs, 1) size
            self.W = np.random.random((n_inputs, 1))

        # Add bias input to the array
        X = np.c_[np.ones(n_samples), X]

        for i in range(self.iteractions):
            for j, x in enumerate(X):
                x = np.transpose([x])         # x is transposed because it should be a column
                y_real = np.transpose([Y[j]]) # y is transposed because it should be a column
                
                v = np.dot(np.transpose(self.W), x)         # v = w^T*x
                y_predicted = self.sigmoid_func(v)          # y = f(v)
                delta = y_real - y_predicted

                correction = self.eta * np.dot(x, np.transpose(delta))
                self.W += correction
        
    def classify(self, X):
        N, _ = X.shape
        v = np.dot(np.transpose(self.W),np.transpose(np.c_[np.ones(N),X]))
        y_predicted = self.sigmoid_func(v)
        return np.transpose(y_predicted)

    def predict(self, X):
        Y = self.classify(X)
        class_predicted = []
        if self.number_of_classes > 2:
            for y in Y:
                class_predicted.append(np.argmax(y))
        else:
            for y in Y:
                class_predicted.append(1) if y > 0.5 else class_predicted.append(0)

        # Returns the index (which represents the class) of the biggest values
        return class_predicted