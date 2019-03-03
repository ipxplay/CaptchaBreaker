import numpy as np


class Perceptron:
    # initialize weight matrix and store the alpha value(learning rate)
    def __init__(self, N, alpha):
        self.W = np.random.randn(N + 1) / np.sqrt(N)
        self.alpha = alpha

    def step(self, x):
        return 1 if x > 0 else 0

    def fit(self, X, y, epochs=10):
        X = np.c_[X, np.ones(X.shape[0])]
        for _ in np.arange(0, epochs):
            for (x, target) in zip(X, y):
                # take the dot product between the input features
                # and the weight matrix, then pass this value
                # through the step function to obtain the prediction
                p = self.step(np.dot(x, self.W))
                if p != target:
                    error = p - target
                    self.W += -self.alpha * error * x

    def predict(self, X, addBias=True):
        X = np.atleast_2d(X)

        if addBias:
            X = np.c_[X, np.ones(X.shape[0])]

        return self.step(np.dot(X, self.W))
