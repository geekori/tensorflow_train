# 数字图像分类（MNIST）

import sys,os
from dataset.mnist import load_mnist

'''
[5,4]

[[0. 0.]
 [0. 0.]
 [0. 0.]
 [0. 0.]
 [0. 1.]
 [1. 0.]]
'''
(x_train,t_train),(x_test,t_test) = load_mnist(flatten=True,normalize=False)

print(x_train.shape)
print(x_test.shape)

import numpy as np
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

img = x_train[10]
label = t_train[10]
print(label)
print(img.shape)
img = img.reshape(28,28)
print(img.shape)
img_show(img)

import pickle
# 初始化神经网络
def init_network():
    with open('weight.pkl','rb') as f:
        network = pickle.load(f)
    return network

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
def predict(network, x):
    W1,W2,W3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']

    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1) # 第一层的输出

    a2 = np.dot(z1,W2) + b2
    z2 = sigmoid(a2)  # 第二次的输出

    a3 = np.dot(z2,W3) + b3
    y = softmax(a3)
    return y

network = init_network()

accuracy_count = 0
print(np.argmax(predict(network, x_test[10])))

label = t_test[10]
print(label)


for i in range(len(x_test)):
    y = predict(network, x_test[i])
    p = np.argmax(y)
    if p == t_test[i]:
        accuracy_count += 1
print('Accuracy:' + str(float(accuracy_count)/len(x_test)),accuracy_count)









