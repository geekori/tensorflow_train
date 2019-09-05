import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

y_array = tf.linspace(-3.,5.,500)
target = tf.constant(1.)
targets = tf.fill([500,],1.)

# Huber损失函数
def huber_loss(y,t):
    return tf.maximum(0.,1. - tf.multiply(t,y)).numpy()
out1 = huber_loss(y_array,targets)
print(out1)

# Cross entropy loss

def cross_entropy_loss(y,t):
    return -tf.multiply(t,tf.math.log(y)) - tf.multiply((1.-t),tf.math.log(1.-y))

out2 = cross_entropy_loss(y_array,targets)
print(out2)

# Cross entropy sigmoid loss
def cross_entropy_sigmoid_loss(y,t):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits = y_array,labels=targets)

out3 = cross_entropy_sigmoid_loss(y_array,targets)

# weight cross entropy
def weight_cross_entropy_loss(y,t,weight):
    return tf.nn.weighted_cross_entropy_with_logits(labels=targets,logits=y_array,pos_weight=weight)
out4 = weight_cross_entropy_loss(y_array,targets,0.5)

import matplotlib.pyplot as plt
plt.plot(y_array,out1,'b-',label='Huber loss')
plt.plot(y_array,out2,'r--',label='Cross entropy loss')
plt.plot(y_array,out3,'g-',label='Cross entropy sigmoid Loss')
plt.plot(y_array,out4,'k:',label='Weight cross entropy loss')
plt.ylim(-1.5,3)
plt.legend(loc='lower right',prop={'size':12})
plt.grid()
plt.show()


