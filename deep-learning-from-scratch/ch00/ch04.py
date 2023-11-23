import numpy as np
import pickle
import sys, os, time
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient
#from two_layer_net import TwoLayerNet
import matplotlib.pylab as plt

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)
    

def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

def relu_function(x):
    return np.maximum(0, x)

def draw_graph():
    x = np.arange(-5.0, 5.0, 0.1)
    y = step_function(x)
    plt.plot(x,y)
    z = sigmoid_function(x)
    plt.plot(x,z)
    w = relu_function(x)
    plt.plot(x,w)
    plt.ylim(-0.1, 5.0)
    plt.show()

def identity_function(x):
    return x

#def init_network():
#    network = {}
#    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
#    network['b1'] = np.array([0.1, 0.2, 0.3])
#    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
#    network['b2'] = np.array([0.1, 0.2])
#    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
#    network['b3'] = np.array([0.1, 0.2])

#    return network

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3'] 
    b1, b2, b3 = network['b1'], network['b2'], network['b3'] 

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid_function(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid_function(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3'] 
    b1, b2, b3 = network['b1'], network['b2'], network['b3'] 

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
    
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        dz1 = np.dot(dy, W2.T)
        da1 = sigmoid_grad(a1) * dz1
        grads['W1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)

        return grads

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
    
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y
        
    # x:入力データ, t:教師データ
    def loss(self, x, t):
        y = self.predict(x)
        
        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x:入力データ, t:教師データ
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
        
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        dz1 = np.dot(dy, W2.T)
        da1 = sigmoid_grad(a1) * dz1
        grads['W1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)

        return grads


#---------------------------------------------------
def main8():     # ミニバッチ学習の実装 (結果の確認）
    with open('train_loss_list.pkl', 'rb') as f:
        train_loss_list = pickle.load(f)
#    x = np.arange(len(train_loss_list))
    x = np.arange(1000)
    plt.plot(x, train_loss_list[:1000])
    plt.plot(x, train_loss_list[9000:])
    plt.ylim(0., 10.0)
    plt.show()


def main():     # ミニバッチ学習の実装
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    train_loss_list = []

    # Hyper parameters
    iters_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1

    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
   
    for i in range(iters_num):
        if i % 10 == 0: pass
    #        print('i:', i)
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
    
        # 勾配の計算
        # grad = network.numerical_gradient(x_batch, t_batch)
        grad = network.gradient(x_batch, t_batch)

        # Parameters update
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]
        
        # 学習経過の記録
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)
    
#    print('train_loss_list:', train_loss_list)
    with open('train_loss_list.pkl', 'wb') as f:
        pickle.dump(train_loss_list,f)

    x = np.arange(1000)
    plt.plot(x, train_loss_list[:1000])
    plt.plot(x, train_loss_list[9000:])
    plt.ylim(0., 10.)
    plt.show()


def main6():    # 2層ニューラルネットワークのクラス
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
    print(batch_mask[:20])
    print(batch_mask[-20:])
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    print(x_batch)
    print(t_batch)
if __name__ == '__main__':
    start_time = time.time()
    print('Start:', start_time)
    main()
    end_time = time.time()
    print('Start: {:.0f} (sec)'.format(start_time))
    print('End:: {:.0f} (sec)'.format(end_time))
    print('E-S: {:.2f} (sec)'.format(end_time-start_time))
