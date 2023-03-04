import numpy as np
import pandas as pd


class LinearRegression:
    def __init__(self, learning_rate=0.001, n_iter=1000000, tolerance=0.000001) -> None:
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None
        self.tolerance = tolerance
        self.gradients = []

    def gradient_descent(self, X, Y):
        X = X.values
        Y = Y.values
        # print(type(X))
        # pick random weights and bias

        self.weights = np.random.randn(X.shape[1], 1)  # rand num between number of col and 1
        self.bias = np.random.randn(1)  # rand num between 0 and 1
        vector = [self.bias] + [weight for weight in self.weights]

        for i in range(self.n_iter):
            # make prediction
            diff = -self.learning_rate * np.array(self.ssr_gradient(X, Y, vector))

            if np.all(np.abs(diff[0]) <= self.tolerance):
                break

            vector += diff

        self.weights = vector[1:]
        self.bias = vector[0]
        return vector

    def ssr_gradient(self, X, Y, w):  # gradiant function

        grad = [0] * (X.shape[1] + 1)

        for i in range(X.shape[0]):
            x, y = X[i], Y[i]
            res = w[0] + np.dot(x, w[1:]) - y

            grad[0] += res
            for i in range(1, len(grad)):
                grad[i] += res * x[i - 1]
        return [g / X.shape[0] for g in grad]

    def predict(self, X):
        # print(X)
        # X = np.array(list(map(float, X)))
        weights_reshaped = np.reshape(self.weights, (X.shape[1], 1))
        Y_pred = self.bias + np.dot(X, weights_reshaped)
        Y_pred = np.squeeze(Y_pred)
        return Y_pred