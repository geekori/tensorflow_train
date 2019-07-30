# 用偏置代替阈值

import numpy as np
x = np.array([0,1])
x

w = np.array([0.5,0.5])

b = -0.8

print(w * x)
# 计算y
print(np.sum(w * x) + b)