def f(x):
    return x**2 + 2

import numpy as np
import matplotlib.pylab as plt

x = np.arange(0.0,20.0,0.1)

y = f(x)

plt.xlabel('x')
plt.ylabel('f(x)')
plt.plot(x,y)
plt.show()

def derivative1(f,x):
    h = 10e-50
    return (f(x+h)-f(x))/h
def derivative2(f,x):
    h = 1e-4
    return (f(x+h)-f(x))/h
def derivative3(f,x):
    h = 1e-4
    return(f(x+h) - f(x-h)) / (2 * h)
print(derivative1(f,10))
print(derivative2(f,10))  # 20  x^2 + 2  x^n  导数为n * x ^(n-1)  2x
print(derivative2(f,3))  # 6
print(derivative3(f,3))



