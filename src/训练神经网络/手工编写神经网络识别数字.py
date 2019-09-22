'''
激活函数：决定了神经网络中流淌的是什么样的数据

损失函数：神经网络学习的目的是让损失函数的值尽可能小，当损失函数的值达到我们的预期，
        就将权重（W）和偏置（b）保存，这样就训练好了一个神经网络模型

梯度、梯度下降

损失函数的自变量是整个神经网络的所有权重和偏置

mini-batch

从创建神经网络、到训练神经网络，再到使用训练好的神经网络识别数字图像的步骤

1. 从训练数据中随机选择一批数据，称为mini-batch。目的是减少mini-batch的损失函数的平均值
mini-batch = 100

2. 需要求出每一个权重参数（W）和偏置（b）的梯度，梯度表示损失函数的值减小最多的方向

3. 将权重和偏置延梯度方向进行微小更新

4. 重复步骤1


10000

5. 当完成训练后，就会产生一组符合要求的W和b，将W和b保存，就会形成一个训练完的神经网络

6. 将要识别的数字图像像素的数据作为输入，然后神经网络就会得到输出


MNIST   70000张数字图像   28*28 = 784

'''

# coding: utf-8
import sys, os
import numpy as np
# 交叉熵误差损失函数
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)
# 用于分类任务的激活函数
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x) # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))
# 用于神经网络内部节点的激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'])   # 以灵活的方式访问多维数组，默认是按行访问
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 还原值
        it.iternext()

    return grad

# 两层神经网络

class TwoLayterNet:
    def __init__(self,input_size,hidden_size,output_size):
        # W和b的初始值
        self.params = {}
        weight_init_std = 0.01
        # input_size = 784
        # hiddle_size = 100
        # (784,100)
        # 输入层与隐藏层之间的W
        self.params['W1'] = weight_init_std * np.random.randn(input_size,hidden_size)
        # 箭头指向的神经元节点有多少个，b就有多少个
        self.params['b1'] = np.zeros(hidden_size)
        # 隐藏层与输出层之间的W
        # hiddle_size = 100
        # output_size = 10
        # (100,10)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
    # 用于数字图像分类
    def predict(self,x):
        W1,W2= self.params['W1'],self.params['W2']
        b1,b2 = self.params['b1'],self.params['b2']
        # (784,100) + (100,)
        a1 = np.dot(x,W1) + b1
        z1 = sigmoid(a1)

        a2 = np.dot(z1,W2) + b2
        y = softmax(a2)
        return y
    # 计算平均损失函数值
    def loss(self,x,t):
        y = self.predict(x)
        return cross_entropy_error(y,t)
    # 数值方式计算梯度
    def numerical_gradient(self,x,t):
        loss_W = lambda  W:self.loss(x,t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W,self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads
    # 误差反向传播计算梯度
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]


        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        dy = (y - t) / batch_num

        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)


        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        return grads

'''
a = np.array([[2,3,4],[5,4,3]])
b = np.array([7,8,9])
# (2,3)+(3,)
c = a + b
print(a)
print(b)

print(c)
'''

net = TwoLayterNet(input_size=784,hidden_size=100,output_size=10)
x = np.random.rand(100,784)
y = net.predict(x)
#print(y)
# 60000张图片用于训练   10000图片用于测试
from dataset.mnist import load_mnist

(x_train,t_train),(x_test,t_test) = load_mnist(normalize=True,one_hot_label=True)
#print(x_train)
#print(t_train)

train_loss_list = []

# 超参数
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1  # 学习率

for i in range(iters_num):
    # 随机获取100个训练数据
    batch_mask = np.random.choice(train_size,batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 计算梯度（得到了W和b应该加上或减少的幅度）
    grad = net.gradient(x_batch,t_batch)
   # print(grad)
    # 更新参数
    for key in ('W1','b1','W2','b2'):
        net.params[key] -= learning_rate * grad[key]
    loss = net.loss(x_batch,t_batch)

    train_loss_list.append(loss)

print(train_loss_list)
# 可以将net.params保存，下次可以直接装载W和b
print(net.params)

x = np.arange(1,10001,1)

import matplotlib.pylab as plt


plt.xlabel('iteration')
plt.ylabel('loss')
plt.figure()
plt.subplot(2,2,1)
plt.plot(x,train_loss_list)
plt.subplot(2,2,2)
plt.imshow(x_test[3:4].reshape(28,28))
plt.subplot(2,2,3)
plt.imshow(x_test[1000:1001].reshape(28,28))

plt.subplot(2,2,4)
plt.imshow(x_test[5000:5001].reshape(28,28))

plt.show()

# 1000次迭代，无法正确分类（将9识别为7），10000次迭代，成功进行分类
print(np.argmax(net.predict(x_test[1000:1001])))
print(np.argmax(t_test[1000:1001]))















