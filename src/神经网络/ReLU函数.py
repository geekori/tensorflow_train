# ReLU
import numpy as np

def relu(x):
    return np.maximum(0,x)
print(relu(20))
print(relu(-20))

import matplotlib.pylab as plt

x = np.arange(-5.0,5.0,0.1) # 根据间隔，返回相应数量的值
print(x)
y = relu(x)

plt.plot(x,y)
plt.show()
