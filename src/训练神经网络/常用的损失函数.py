import numpy as np

y1 = [0.1,0.05,0.6,00,0.05,0.1,0.0,0.1,0.0,0.0]
t1 = [0,0,1,0,0,0,0,0,0,0]

y2 = [0.1,0.05,0.1,00,0.05,0.1,0.0,0.6,0.0,0.0]
t2 = [0,0,1,0,0,0,0,0,0,0]

# 均方误差
def mean_squared_error(y,t):
    n = len(y)
    return (np.sum((np.array(y) - np.array(t))**2)) / n

print(mean_squared_error(y1,t1))
print(mean_squared_error(y2,t2))

# 交叉熵误差
def cross_entropy_error(y,t):
    delta = 1e-7
    return -np.sum(np.array(t) * np.log(np.array(y) + delta))

print(cross_entropy_error(y1,t1))
print(cross_entropy_error(y2,t2))
