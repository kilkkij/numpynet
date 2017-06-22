
import numpy as np
from numpy import tanh, exp, empty, zeros, zeros_like, empty_like, dot, asarray, copy, einsum
from numpy.random import rand, seed

try:
    from numexpr import evaluate
except ImportError:
    K, dK = lambda x: 1.7159*tanh(0.666*x), lambda x: 1.1427894*(1-(tanh(0.666*x))**2)
else:
    K, dK = lambda x: evaluate('1.7159*tanh(0.666*x)'), lambda x: evaluate('1.1427894*(1-(tanh(0.666*x))**2)')

REGULARIZATION_PARAMETER = 1.e-4
REGULARIZATION_EXPONENT = 2

class Net(object):

    """
    Data attributes:
    W       list of weight arrays (obs-size, next-layer-size, layer-size)
    x       list of unactivated data arrays (obs-size, layer-size, 1)
    ax      list of activated data arrays (obs-size, layer-size, 1)
    """

    def __init__(self, hidden_sizes, data):
        # Data access
        self.data = data
        # Shape of net
        assert(len(hidden_sizes) >= 1)
        self.form = (data.insize,) + tuple(hidden_sizes) + (data.outsize,)
        self.depth = len(self.form)
        # Data arrays
        self.ax = [empty((data.batchsize, size)) for size in self.form]
        # Data arrays but unactivated
        self.x = [None] + [empty_like(xi) for xi in self.ax[1:]]
        # Gradients of energy wrt. unactivated data at each layer
        self.dEdx = [None] + [empty_like(xi) for xi in self.ax[1:]]
        # Uniform random for weight initialization
        unif_rand = lambda shape: 2*(2*rand(*shape) - 1)
        # Weights
        # self.W = [unif_rand((data.batchsize, self.form[i+1], self.form[i])) for i in range(self.depth-1)]
        self.W = [unif_rand((self.form[i+1], self.form[i])) for i in range(self.depth-1)]
        # Weight gradients
        self.dEdW = [zeros((data.batchsize,)+Wi.shape) for Wi in self.W]
        # First data array equals input observations
        self.ax[0] = asarray([yi for yi, yo in data.obs])

    def activate(self):
        x, ax, W = self.x, self.ax, self.W
        for i in range(1, self.depth):
            x[i][:] = einsum('ij,dj->di', W[i-1], ax[i-1])
            ax[i][:] = K(x[i])

    def weight_gradient(self):

        x, ax, W, dEdx, dEdW = self.x, self.ax, self.W, self.dEdx, self.dEdW

        # Output layer first.
        # Observation error
        error = ax[-1] - self.data.values
        # E for energy = cost function = log-inv-prob, d for derivative
        dEdx[-1] = error*dK(x[-1])
        dEdW[-1] = einsum('ij,ik->ijk', dEdx[-1], ax[-2])

        # The rest of the weight layers.
        # For 1 hidden layer, this loop is just i=0
        for i in range(self.depth-3, -1, -1):
            # Gradient wrt. unactivated data.
            # The sum corresponds to contributions by (unactivated) i+2 data layer and i+1 weight layer.
            # To match dimensions, shape of the sum expression is spread.
            # Each activation gradient is multiplied elementwise.
            dEdx_W = einsum('dn,ni->di', dEdx[i+2], W[i+1])
            dEdx[i+1][:] = dEdx_W*dK(x[i+1])
            # Gradient wrt. weight matrix
            dEdW[i][:] = einsum('dn,di->dni', dEdx[i+1], ax[i])
            # Regularization term gradient
            reg_grad = W[i]**(REGULARIZATION_EXPONENT-1)*REGULARIZATION_PARAMETER
            dEdW[i][:] += reg_grad[None, ...]

        return dEdW