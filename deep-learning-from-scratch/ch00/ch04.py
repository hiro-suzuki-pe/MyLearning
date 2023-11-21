import numpy as np
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from common.functions import softmax
from common.gradient import numerical_gradient

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

def cross_entropy_error(y, t):
    delta = 1e-7
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + delta)) / batch_size

def numerical_diff(f, x):
    h = 1e-4 
    return (f(x+h) - f(x-h)) / (2*h)

def numerical_gradient(f, x):
    h = 1e-4 
    grad = np.zeros_like(x)
    for idx in range(x.shape[0]):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3) # initialize with Gaussian dist.

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss    

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        print('self.params[W1]:', self.params['W1'].shape)
        print('self.params[b1]:', self.params['b1'].shape)
        print('self.params[W2]:', self.params['W2'].shape)
        print('self.params[b2]:', self.params['b2'].shape)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y
    
    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads
    


#---------------------------------------------------
def main():
    net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
#    net = TwoLayerNet(784, 100, 10)
    print(net.params['W1'].shape)
    print(net.params['b1'].shape)
    print(net.params['W2'].shape)
    print(net.params['b2'].shape)

    x = np.random.rand(100, 784)
    y = net.predict(x)

    x = np.random.rand(100, 784)
    t = np.random.rand(100, 10)

    grads = net.numerical_gradient(x, t)
    print(grads['W1'].shape)
    print(grads['b1'].shape)
    print(grads['W2'].shape)
    print(grads['b2'].shape)


def main5():
    net = simpleNet()

    x = np.array([0.6, 0.9])
    p = net.predict(x)
    print(p)


def func(x):
        return np.sum(x**2)
    
def main4():
    init_x = np.array([100, 50, 70, 30], dtype=float)
    conv = gradient_descent(func, init_x, 0.1, 200)
    print('gradient_descent(func, init_x, 0.1, 200)', conv)

def main3():


    print('numerical_diff(sin,0):', numerical_diff(np.sin, 0))
    print('numerical_diff(cos,0):', numerical_diff(np.cos, 0))

    x = np.array([1., 2., 3.])
    print('numerical_gradient(x**2):', numerical_gradient(func, x))

def main2():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    print(x_train.shape)
    print(t_train.shape)
    train_size = x_train.shape[0]
    batch_size = 10
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    print('batch_mask:', batch_mask)
#    print('t_batch:', t_batch)

def main1():
    y = np.array([1.2, 3.5, 4.3])
    t = np.array([1, 3, 4])
    mse = mean_squared_error(y, t)
    print('mean_squared_error:', mse)

    y = np.array([1.2, 0.5, 4.3])
    t = np.array([0, 1, 0])

    cee = cross_entropy_error(y, t)
    print('cross_entropy_error:', cee)


#---------------------------------------------------
if __name__ == '__main__':
    main()