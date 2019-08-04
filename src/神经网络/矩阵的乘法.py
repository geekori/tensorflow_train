import numpy as np

a = np.array([[1,2],[3,4],[5,6]])
print(a.shape)

b = np.array([[5,6,7],[7,8,9]])
print(b.shape)

c = np.dot(a,b)
print(c)

d = np.array([[5,6,7,8],[6,8,5,3]])
print(d.shape)

print(np.dot(a,d))

x = np.array([1,2])
print(x.shape)

print(np.dot(a,x))