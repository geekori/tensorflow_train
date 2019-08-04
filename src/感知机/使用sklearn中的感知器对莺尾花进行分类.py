# 使用sklearn中的感知器对莺尾花进行分类

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']

iris = load_iris()
X = iris.data[:,(2,3)]  # 花瓣长度，花瓣宽度
print(X)

y = (iris.target == 0).astype(np.int)
print(y)
per_clf = Perceptron(max_iter=100,random_state=42)
per_clf.fit(X,y)
y_pred = per_clf.predict([[2,1.2]])
print(y_pred)

#######
axes = [0,5, 0,2]
x0,x1 = np.meshgrid(
    np.linspace(axes[0],axes[1],500),
    np.linspace(axes[2],axes[3],200)
)

X_new = np.c_[x0.ravel(),x1.ravel()]
print(X_new)
y_predict = per_clf.predict(X_new)
print(y_predict)

zz = y_predict.reshape(x0.shape)
print(zz)

plt.figure(figsize = (10,4))
plt.plot(X[y == 0,0],X[y == 0, 1],'bs',label = "Not Iris-Setosa")
plt.plot(X[y == 1,0],X[y == 1,1],'ro',label = "Iron-Setosa")

from matplotlib.colors import ListedColormap
custom_cmap = ListedColormap(['#9899ff','#fafab0'])

plt.contourf(x0,x1,zz,cmap=custom_cmap)

plt.xlabel('花瓣长度',fontsize=14)
plt.ylabel('花瓣宽度',fontsize=14)

plt.axis(axes)

plt.show()
