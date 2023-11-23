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

    print('W1.shape', W1.shape)
    print('W2.shape', W2.shape)
    print('W3.shape', W3.shape)
    print('b1.shape', b1.shape)
    print('b2.shape', b2.shape)
    print('b3.shape', b3.shape)
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

def main():
    x, t = get_data()
    network = init_network()

    accuracy_cnt = 0
    for i in range(len(x)):
        y = predict(network, x[i])
        p = np.argmax(y)
        if p == t[i]:
            accuracy_cnt += 1
    print("Accuracy:" + str(float(accuracy_cnt) / len(x)), 'len(x)' + str(int(len(x))))

if __name__ == '__main__':
    main()
