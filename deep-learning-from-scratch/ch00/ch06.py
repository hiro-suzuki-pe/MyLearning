import sys, os, time
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import numpy as np
import matplotlib.pylab as plt
#from common.layers import *
from collections import OrderedDict
import pickle

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.memontum * self.v[key] - self.lr*grads[key]
            params[key] += self.v[key]

class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7
                                                   )
   
#---------------------------------
def main():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    network = TwolayerNet(input_size=784, hidden_size=50, output_size=10)
    x_batch = x_train[:30]
    t_batch = t_train[:30]
    
    grad_numerical = network.numerical_gradient(x_batch, t_batch)
    grad_backprop = network.gradient(x_batch, t_batch)

    for key in grad_numerical.keys():
        diff = np.average(np.abs(grad_backprop[key]-grad_numerical[key]))
        print(key + ':' + str(diff))
#        print(key + ':' + str(diff), grad_backprop[key]-grad_numerical[key])


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print('\n\nStart: {:.0f} (sec)'.format(start_time))
    print('End:: {:.0f} (sec)'.format(end_time))
    print('E-S: {:.2f} (sec)'.format(end_time-start_time))
