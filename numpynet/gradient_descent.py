
import numpy as np


class Descender(object):
    """Gradient descent with momentum."""
    

    def __init__(self, net, lrate, momentum):
        """
        lrate       Learning rate. E.g. lrate=0.2
        momentum    Momentum for faster convergenge. E.g. 0.4
        """
        self.net = net
        self.lrate = lrate
        self.momentum = momentum
        self.delta_prev = [np.copy(self.reduce(dWi)) for dWi in net.dEdW]

    def reduce(self, delta):
        return np.mean(delta, axis=0)

    def descend_batch(self):
        W = self.net.W
        ax = self.net.ax
        # Activate net
        self.net.activate()
        # Energy gradient matrices
        dEdW = self.net.weight_gradient()
        # Update weights
        for wi in range(self.net.depth-1):
            # Weight increment. Doesn't include momentum
            delta_wi = -self.lrate*dEdW[wi]
            # Reduction over batch axis 
            # because there are multiple inputs at each gradient calculation
            delta_wi = self.reduce(delta_wi)
            # Descend step
            W[wi][:] += delta_wi + self.momentum*self.delta_prev[wi]
            # Save weight increment for the next step
            self.delta_prev[wi][:] = delta_wi
        # Squared errors
        sqerrors = (self.net.data.output - ax[-1])**2
        return sqerrors

    def descend(self, N):
        for i in range(N):
            chunk_errors = [self.descend_batch() for _ in range(1)]
            sqerror = np.mean(chunk_errors)
            # print(i, sqerror)
            yield i, sqerror