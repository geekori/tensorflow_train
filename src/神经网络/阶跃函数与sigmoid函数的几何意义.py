# 阶跃函数与sigmoid函数的几何意义
import numpy as np
# 阶跃函数
def jump_func1(x):
    if x > 0:
        return 1
    else:
        return 0

print(jump_func1(0.6))

def jump_func2(x):
    y = x > 0
    return y.astype(np.int)

x = np.array([-1.0,0.5,1.5])
print(jump_func2(x))

import matplotlib.pylab as plt

x = np.arange(-5.0,5.0,0.1) # 根据间隔，返回相应数量的值
print(x)
y = jump_func2(x)
print(y)
plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()

def sigmoid(x):
    return 1/(1 + np.exp(-x))

print(sigmoid(x))

y1 = jump_func2(x)
y2 = sigmoid(x)
plt.plot(x,y1)
plt.plot(x,y2)
plt.ylim(-0.1,1.1)
plt.show()