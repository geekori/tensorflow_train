# 多层感知机
# 与门感知机
def and_perceptron(x1,x2):
    n = x1 * 0.5 + x2 * 0.5
    theta = 0.6
    if n <= theta:
        return 0
    elif n > theta:
        return 1

# 或门感知机
def or_perceptron(x1, x2):
    n = x1 * 0.8 + x2 * 0.7
    theta = 0.6

    if n <= theta:
        return 0
    elif n > theta:
        return 1

# 与非门感知机
def and_not_perceptron(x1, x2):
    n = -0.8 * x1 + -0.7 * x2
    theta = -0.9
    if n <= theta:
        return 0
    elif n > theta:
        return 1

# 异或门感知机
def xor_perceptron(x1, x2):
    s1 = and_not_perceptron(x1, x2)
    s2 = or_perceptron(x1, x2)
    y = and_perceptron(s1, s2)
    return y

print('异或门感知机：x1 = 0, x2 = 0， y = ', xor_perceptron(0,0))
print('异或门感知机：x1 = 1, x2 = 0， y = ', xor_perceptron(1,0))
print('异或门感知机：x1 = 0, x2 = 1， y = ', xor_perceptron(0,1))
print('异或门感知机：x1 = 1, x2 = 1， y = ', xor_perceptron(1,1))