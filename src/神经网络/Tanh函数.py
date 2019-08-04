# Tanh
import numpy as np

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
print(tanh(20))
print(tanh(-20))

import matplotlib.pylab as plt

x = np.arange(-5.0,5.0,0.1) # 根据间隔，返回相应数量的值
print(x)
y = tanh(x)

plt.plot(x,y)
plt.show()
