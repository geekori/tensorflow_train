# 三层神经网络的完整实现

import numpy as np

def init_network():
    network = {}
    network['W1'] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1'] = np.array([0.1,0.2,0.3])
    network['W2'] = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['b2'] = np.array([0.1,0.2])
    network['W3'] = np.array([[0.1,0.3],[0.2,0.4]])
    network['b3'] = np.array([0.1,0.2])
    return network

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def forward(network,X,level,activation):
    W = network['W' + str(level)]
    b = network['b' + str(level)]
    a = np.dot(X,W) + b
    z = activation(a)
    return z
network = init_network()
X = np.array([1.0,0.5])
for i in range(1,4):
    X = forward(network, X,i,tanh)
print(X)

