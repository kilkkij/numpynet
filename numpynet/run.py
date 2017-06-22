
from data import ArrayData
from net import Net
from gradient_descent import Descender

if __name__=='__main__':

    xor_obs = [([0.0, 0.0], [-1.]), ([0.0, 1.0], [1.0]), ([1.0, 0.0], [1.0]), ([1.0, 1.0], [-1.]),]
    xor_obs = xor_obs*100000
    xordata = ArrayData(xor_obs)

    net = Net([5], xordata)
    desc = Descender(net, .2, .4)

    idx, sqes = [], []
    try:
        for i, s in desc.descend(100):
        	print(i, s)
        	idx.append(i)
        	sqes.append(s)
    except KeyboardInterrupt:
        pass

    from pylab import *
    ion()
    plot(idx, sqes)
    ylim([0, sqes[0]*1.05])
