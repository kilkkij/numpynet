
"""
Currently only XOR data available.
"""

from numpy import asarray

obs = [([0.0, 0.0], [-1.]), ([0.0, 1.0], [1.0]), ([1.0, 0.0], [1.0]), ([1.0, 1.0], [-1.]),]
obs = obs*100000

class ArrayData:
	"""XOR inputs and observations."""
	def __init__(self, datapairs):
		assert(len(datapairs))
		i, o = zip(*datapairs)
		self.input = i
		self.output = o
		self.batchsize = len(datapairs)
		self.insize = len(self.input[0])
		self.outsize = len(self.output[0])
