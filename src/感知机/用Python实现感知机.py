# 用Python实现感知机

and_params = {"w1":0.5,"w2":0.5,'theta':0.6}
or_params = {"w1":0.8,"w2":0.7,"theta":0.6}
and_not_params = {"w1":-0.8,"w2":-0.7,"theta":-0.9}

# 感知机函数
def perceptron(x1,x2,params):
    n = x1 * params["w1"] + x2 * params["w2"]
    theta = params["theta"]
    if n <= theta:
        return 0
    elif n > theta:
        return 1

print('与门电路感知机：x1 = 0,x2 = 0,y = ',perceptron(0, 0, and_params))
print('与门电路感知机：x1 = 1,x2 = 0,y = ',perceptron(1, 0, and_params))
print('与门电路感知机：x1 = 0,x2 = 1,y = ',perceptron(0, 1, and_params))
print('与门电路感知机：x1 = 1,x2 = 1,y = ',perceptron(1, 1, and_params))


print('或门电路感知机：x1 = 0,x2 = 0,y = ',perceptron(0, 0, or_params))
print('或门电路感知机：x1 = 1,x2 = 0,y = ',perceptron(1, 0, or_params))
print('或门电路感知机：x1 = 0,x2 = 1,y = ',perceptron(0, 1, or_params))
print('或门电路感知机：x1 = 1,x2 = 1,y = ',perceptron(1, 1, or_params))

print('与非门电路感知机：x1 = 0,x2 = 0,y = ',perceptron(0, 0, and_not_params))
print('与非门电路感知机：x1 = 1,x2 = 0,y = ',perceptron(1, 0, and_not_params))
print('与非门电路感知机：x1 = 0,x2 = 1,y = ',perceptron(0, 1, and_not_params))
print('与非门电路感知机：x1 = 1,x2 = 1,y = ',perceptron(1, 1, and_not_params))
