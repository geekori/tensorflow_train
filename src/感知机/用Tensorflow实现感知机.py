# 用Tensorflow实现感知机

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

and_params = {'w1':tf.Variable(0.5),'w2':tf.Variable(0.5),'theta':tf.Variable(0.6)}
or_params = {'w1':tf.Variable(0.8),'w2':tf.Variable(0.7),'theta':tf.Variable(0.6)}
and_not_params = {'w1':tf.Variable(-0.8),'w2':tf.Variable(-0.7),'theta':tf.Variable(-0.9)}

# 感知机函数
def perceptron(x1,x2,params):
    w1 = params['w1']
    w2 = params['w2']
    theta = params['theta'].numpy()
    n = (x1 * w1 + x2 * w2).numpy()
    if n <= theta:
        return tf.constant(0)
    elif n > theta:
        return tf.constant(1)

x1 = tf.Variable(1.0)
x2 = tf.Variable(1.0)
y = perceptron(x1,x2,and_params)
print("与门电路感知机：x1 = 1, x2 = 0, y = ",y.numpy())


x1 = tf.Variable(0.0)
x2 = tf.Variable(1.0)
y = perceptron(x1,x2,and_params)
print("与门电路感知机：x1 = 0, x2 = 1, y = ",y.numpy())

x1 = tf.Variable(1.0)
x2 = tf.Variable(1.0)
y = perceptron(x1,x2,and_not_params)
print("与非门电路感知机：x1 = 1, x2 = 1, y = ",y.numpy())