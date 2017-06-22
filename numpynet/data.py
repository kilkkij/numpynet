
"""
"""

from numpy import asarray

class ArrayData:
	"""Inputs (that match input layer) and observations (that match output layer).
    For example, XOR data:
    [([0.0, 0.0], [-1.]), ([0.0, 1.0], [1.0]), ([1.0, 0.0], [1.0]), ([1.0, 1.0], [-1.])]
    """
	def __init__(self, datapairs):
		assert(len(datapairs))
		i, o = zip(*datapairs)
		self.input = i
		self.output = o
		self.batchsize = len(datapairs)
		self.insize = len(self.input[0])
		self.outsize = len(self.output[0])
