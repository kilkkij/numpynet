
import numpy as np
import unittest
import sys
sys.path.append('../')
from net import Net
from data import ArrayData
from gradient_descent import Descender

xor_data = [([0.0, 0.0], [-1.]), ([0.0, 1.0], [1.0]), ([1.0, 0.0], [1.0]), ([1.0, 1.0], [-1.]),]

class DescenderTests(unittest.TestCase):

    def setUp(self):
        self.data = ArrayData(xor_data)
        self.net = Net([3], self.data)
        self.desc = Descender(self.net, .2, .4)

    def test_descent_convergence(self):
        keys, errors = list(zip(*self.desc.descend(30)))
        self.assertTrue(errors[-1] < errors[0])

if __name__=="__main__":
    unittest.main()