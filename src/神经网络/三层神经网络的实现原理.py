import numpy as np

X = np.array([1.0,0.5])
W1 = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
B1 = np.array([0.1,0.2,0.3])
print(X.shape)
print(W1.shape)
print(B1.shape)

A1 = np.dot(X,W1) + B1
print(A1)

def sigmoid(x):
    return 1/(1 + np.exp(-x))
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
Z1 = sigmoid(A1)
Z1 = tanh(A1)
print(Z1)

# 处理第2层神经元（行：输入数量， 列：神经元数量）
W2 = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
B2 = np.array([0.1,0.2])
A2 = np.dot(Z1,W2) + B2
print(A2)

Z2 = tanh(A2)
print(Z2)

# 处理输出层
W3 = np.array([[0.1,0.3],[0.2,0.4]])
B3 = np.array([0.1,0.2])
A3 = np.dot(Z2,W3) + B3
def identity_func(x):
    return x
print(A3)
Y= identity_func(A3)
print(Y)






