# 用Python和Tensorflow实现带偏置的感知机

import numpy as np

def and_perceptron(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.8
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
print('与门电路感知机：x1 = 1, x2 = 1,y = ', and_perceptron(1,1))

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def and_perceptron_tf(x1,x2):
    w1 = tf.constant(0.5)
    w2 = tf.constant(0.5)
    b = tf.constant(-0.8)

    tmp = (w1 * x1 + w2 * x2 + b).numpy()
    if tmp <= 0:
        return tf.constant(0)
    else:
        return tf.constant(1)


def or_perceptron_tf(x1, x2):
    w1 = tf.constant(0.8)
    w2 = tf.constant(0.7)
    b = tf.constant(-0.6)

    tmp = (w1 * x1 + w2 * x2 + b).numpy()
    if tmp <= 0:
        return tf.constant(0)
    else:
        return tf.constant(1)
print('与门电路感知机：x1 = 0,x2 = 1,y = ',and_perceptron_tf(tf.constant(0.0),tf.constant(1.0)).numpy())
print('或门电路感知机：x1 = 1,x2 = 1,y = ',or_perceptron_tf(tf.constant(1.0),tf.constant(1.0)).numpy())