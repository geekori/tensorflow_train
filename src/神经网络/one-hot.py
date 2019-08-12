import numpy as np
data = [1,3,9,2,5,6]
data1 = [5,4]
data = np.array(data)
print(data)
# eye：根据参数n返回n*n的矩阵，主对角线为1，其他位置都是0
def convert_to_one_hot(data,C):
    return np.eye(C)[data].T

print(convert_to_one_hot(data,10))