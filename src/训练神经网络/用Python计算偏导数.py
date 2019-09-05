# x0 ^ 2 + x1 ^ 2  = 2 * x0
# 2 * x1
def f(x):
    return x[0] ** 2 + x[1] ** 2

# x0 = 5.0, x1 = 0.9
# x0的偏导数
def f1(x0):
    return x0 ** 2 + 9.0 ** 2

def derivative(f,x):
    h = 1e-4
    return (f(x+h)-f(x))/h

print(derivative(f1,5.0))

# x1的偏导数
def f2(x1):
    return 5.0 ** 2 + x1 ** 2

print(derivative(f2,9.0))
