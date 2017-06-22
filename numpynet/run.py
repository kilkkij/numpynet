
from data import Data
from net import Net
from gradient_descent import Descender

if __name__=='__main__':

    xordata = ArrayData()
    net = Net([5], xordata)
    desc = Descender(net)

    idx, sqes = [], []
    for i, s in desc.descend(100):
    	print(i, s)
    	idx.append(i)
    	sqes.append(s)

    # from pylab import *
    # ion()
    # plot(idx, sqes)
    # ylim([0, sqes[0]*1.05])
