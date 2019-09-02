import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

y_array = tf.linspace(-1.,1.,500)

target = tf.constant(0.)

def square_loss(y,t):
    return tf.square(y-t).numpy()

out1 = square_loss(y_array,target)
print(out1)

def abs_loss(y,t):
    return tf.abs(y-t).numpy()

out2 = abs_loss(y_array,target)
print(out2)

def pseudo_huber_loss(y,t,d):
    d = tf.constant(d)
    return tf.multiply(tf.square(d),tf.sqrt(1. + tf.square((y-t)/d))-1.).numpy()

out3 = pseudo_huber_loss(y_array,target,0.25)
out4 = pseudo_huber_loss(y_array,target,5.0)

import matplotlib.pyplot as plt

plt.plot(y_array,out1,'b-', label='Mean-Square Loss')
plt.plot(y_array,out2,'r--',label='Mean-Abs Loss')
plt.plot(y_array,out3,'k-.',label='P-Huber Loss(0.25)')
plt.plot(y_array,out4,'g:',label='P-Huber Loss(5.0)')

plt.ylim(-0.2,0.4)
plt.legend(loc='lower right',prop={'size':11})
plt.grid()
plt.show()
