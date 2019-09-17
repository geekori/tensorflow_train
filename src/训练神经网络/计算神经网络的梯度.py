import numpy as np

def cross_entropy_error(y,t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

def numerical_gradient(f,x):
    h = 1e-4
    grad = np.zeros_like(x).reshape(2,3)  # 生成与x形状相同的数组
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            tmp_value = x[i][j]
            x[i][j] = tmp_value + h
            fxh1 = f(x)

            x[i][j] = tmp_value - h
            fxh2 = f(x)
            # 计算偏导数
            grad[i][j] = (fxh1 - fxh2) / (2 * h)
            x[i][j] = tmp_value
    return grad
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

# 实现神经网络
class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)
    def predict(self,x):
        return np.dot(x,self.W)
    def loss(self,x,t):
        z = self.predict(x)  # 进行预测
        y = softmax(z)
        loss = cross_entropy_error(y,t)
        return loss
net =SimpleNet()
print(net.W)
x = np.array([0.6,0.9])
p = net.predict(x)
print(p)

print(np.argmax(p))

t = np.array([0,1,0])
print(net.loss(x,t))

def f(W):
    return net.loss(x,t)

dW = numerical_gradient(f,net.W)

print(dW)

