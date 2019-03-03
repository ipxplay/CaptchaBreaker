import numpy as np


class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        self.W = []
        self.layers = layers
        self.alpha = alpha

        # looping form the index of first layer but stop before
        # we reach the last two layers
        for i in np.arange(0, len(layers) - 2):
            # adding an extra node for the bias
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            self.W.append(w / np.sqrt(layers[i]))

        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))

    def __repr__(self):
        # return the neural network architecture
        return f"NeuralNetwork: {'-'.join(str(l) for l in self.layers)}"

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        return x * (1 - x)

    def fit(self, X, y, epochs=1000, displayUpdate=100):
        X = np.c_[X, np.ones(X.shape[0])]
        for epoch in np.arange(0, epochs):
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)

            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                loss = self.calculate_loss(X, y)
                print(f'[INFO] epoch={epoch+1}, loss={loss:.7f}')

    def fit_partial(self, x, y):
        # construct our list of output activations for each layer
        # as our data point flows through the network; the first
        # activation is a special case -- it's just the input
        # feature vector itself
        A = [np.atleast_2d(x)]

        # FEEDFORWARD
        # loop over the layers in the network
        for layer in np.arange(0, len(self.W)):
            # feed forward the activation at the current layer by
            # taking the dot product between the activation and the
            # weight matrix -- this is called "net input" to the
            # current layer
            net = A[layer].dot(self.W[layer])

            # computing the "out input" is simply applying the
            # nonlinear activation to the net input
            out = self.sigmoid(net)

            # once we have the net output, add it to our list of
            # activations
            A.append(out)

        # BACKPROPAGATION
        # the first phase of backpropagation is to compute the difference
        # between our *prediction*(the final output activation in the
        # activations list) and the true target value
        error = A[-1] - y
        D = [error * self.sigmoid_deriv(A[-1])]

        for layer in np.arange(len(A) - 2, 0, -1):
            delta = D[-1].dot(self.W[layer].T)
            delta *= self.sigmoid_deriv(A[layer])
            D.append(delta)

        D = D[::-1]

        # WEIGHT UPDATE PHASE
        for layer in np.arange(0, len(self.W)):
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

    def predict(self, X, addBias=True):
        p = np.atleast_2d(X)
        if addBias:
            p = np.c_[p, np.ones(p.shape[0])]

        for layer in np.arange(0, len(self.W)):
            p = self.sigmoid(np.dot(p, self.W[layer]))

        return p

    def calculate_loss(self, X, targets):
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias=False)
        loss = 0.25 * np.sum((predictions - targets) ** 2)

        return loss
