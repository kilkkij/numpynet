
import numpy as np
import unittest
import sys
sys.path.append('../')
from net import Net
from data import ArrayData

xor_data = [([0.0, 0.0], [-1.]), ([0.0, 1.0], [1.0]), ([1.0, 0.0], [1.0]), ([1.0, 1.0], [-1.]),]
xor_data = xor_data

class NetTests(unittest.TestCase):

    def setUp(self):
        self.data = ArrayData(xor_data)
        self.net = Net(hidden_sizes=[3], data=self.data)

    def test_depth(self):
        self.assertTrue(len(self.net.W)==2)
        self.assertTrue(len(self.net.dEdx)==3)
        self.assertTrue(len(self.net.x)==3)
        self.assertTrue(len(self.net.ax)==3)

    def test_shape(self):
        self.assertTrue(self.net.W[0].shape == (3, 2))
        self.assertTrue(self.net.W[1].shape == (1, 3))

    def test_first_hidden_layer_weight_application(self):
        for i in range(self.net.depth-1):
            self.net.W[i][:] = 1
        self.net.activate()
        self.assertTrue(np.allclose(self.net.x[1], [[ 0.,  0.,  0.],[ 1.,  1.,  1.],[ 1.,  1.,  1.],[ 2.,  2.,  2.],]))

    def test_first_hidden_layer_activation(self):
        for i in range(self.net.depth-1):
            self.net.W[i][:] = 1
        self.net.activate()
        activated = self.net.ax[1]
        # First data point is zeros, the rest have ones. 
        self.assertTrue(np.allclose(activated[0], 0))
        self.assertTrue(np.all(activated[1:, :] > 0))


if __name__=="__main__":
    unittest.main()