import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import numpy as np
import matplotlib.pylab as plt
import pickle

def step_function(x):
    y = x > 0
    return y.astype(np.int32)

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

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid_function(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid_function(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def get_data2():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_train, t_train, x_test, t_test

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

def main():
    x_train, t_train, x_test, t_test = get_data2()
    print(x_train.shape)    
    print(t_train.shape)    
    print(x_test.shape)    
    print(t_test.shape)    

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
    main()
