import numpy as np
# 计算某一点的梯度
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

print(numerical_gradient(f,np.array([3.0,4.0])))
print(numerical_gradient(f,np.array([0.0,3.0])))
print(numerical_gradient(f,np.array([3.0,0.0])))



