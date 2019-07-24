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
    theta = params['theta']

    n = x1 * w1 + x2 * w2
    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as session:
        init.run()
        n = n.eval()
        theta = session.run(theta)
        if n <= theta:
            return tf.constant(0)
        elif n > theta:
            return tf.constant(1)

with tf.compat.v1.Session() as session:
    x1 = tf.Variable(1.0)
    x2 = tf.Variable(1.0)
    y = perceptron(x1,x2,and_params)
    print('与门电路：x1 = 1, x2 = 1, y = ',session.run(y))

    y = perceptron(x1,x2,and_not_params)
    print('与非门电路：x1 = 1, x2 = 1, y = ',session.run(y))