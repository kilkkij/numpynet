
from numpy import asarray

obs = [([0.0, 0.0], [-1.]), ([0.0, 1.0], [1.0]), ([1.0, 0.0], [1.0]), ([1.0, 1.0], [-1.]),]
obs = obs*10000
BATCHSIZE = len(obs)
INSIZE = 2
OUTSIZE = 1
OBSERVATIONS = asarray([yo for yi, yo in obs]).reshape((BATCHSIZE, 1))

class Data:
	def __init__(self):
		self.values = OBSERVATIONS
		self.batchsize = len(obs)
		self.insize = 2
		self.outsize = 1
		self.obs = obs