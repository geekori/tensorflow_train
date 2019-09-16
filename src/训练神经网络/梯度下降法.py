import numpy as np
def numerical_gradient(f,x):
    h = 1e-4
    grad = np.zeros_like(x)  # 生成与x形状相同的数组
    for i in range(x.size):
        tmp_value = x[i]
        x[i] = tmp_value + h
        fxh1 = f(x)

        x[i] = tmp_value - h
        fxh2 = f(x)
        # 计算偏导数
        grad[i] = (fxh1 - fxh2) / (2 * h)
        x[i] = tmp_value
    return grad

def f(x):
    return x[0]**2 + x[1]**2

# 梯度下降法
def gradient_descent(f,init_x,lr=0.01,step_num = 100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f,x)  # 偏导（梯度）
        x -= lr * grad
    return x

init_x = np.array([-3.0,4.0])

print(gradient_descent(f,init_x,lr=0.1))  #[0,0]


def f1(x):
    return (x[0] + 3)**2 + (x[1] - 2)**2
print(gradient_descent(f1,init_x,lr=0.1)) # [-3.  2.]


def gradient_descent_history(f,init_x,lr=0.01,step_num = 100):
    x = init_x
    x_history = []
    for i in range(step_num):
        x_history.append(x.copy())
        grad = numerical_gradient(f,x)  # 偏导（梯度）
        x -= lr * grad
    return x,np.array(x_history)

import matplotlib.pyplot as plt
x,x_history = gradient_descent_history(f,init_x,lr=0.1,step_num=20)
plt.plot([-5,5],[0,0],'--r')
plt.plot([0,0],[-5,5],'--b')

plt.plot(x_history[:,0],x_history[:,1],'o')

plt.xlim(-3.5,3.5)
plt.ylim(-4.5,4.5)
plt.xlabel('x0')
plt.ylabel('x1')

plt.show()