import numpy as np
from dataset.mnist import load_mnist

(x_train,t_train),(x_test,t_test) = load_mnist(normalize=True,one_hot_label=True)

print(x_train.shape)
print(t_train.shape)

batch_size = 1000
train_size = x_train.shape[0]
batch_mask = np.random.choice(train_size,batch_size)
print(batch_mask)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

print(x_batch.shape)


def cross_entropy_loss(y,t):
    if y.ndim == 1:
        t = t.reshape(1,t.size)
        print(y)
        y = y.reshape(1,y.size)
        print(y)
    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(t * np.log(y + delta)) / batch_size

y1 = [0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
t1 = [0,0,1,0,0,0,0,0,0,0]

y2 = [0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0]
t2 = [0,0,1,0,0,0,0,0,0,0]

y3 = [0.1,0.05,0.1,0.6,0.05,0.1,0.0,0.0,0.0,0.0]
t3 = [0,0,0,1,0,0,0,0,0,0]

y4 = [0.1,0.05,0.1,0.0,0.6,0.1,0.0,0.05,0.0,0.0]
t4 = [0,0,0,0,1,0,0,0,0,0]

print(cross_entropy_loss(np.array(y1),np.array(t1))) # 0.510825457099338
print(cross_entropy_loss(np.array(y2),np.array(t2))) # 2.302584092994546
print(cross_entropy_loss(np.array([y1,y2]),np.array([t1,t2]))) # 1.406704775046942
print(cross_entropy_loss(np.array([y1,y2,y3]),np.array([t1,t2,t3]))) # 1.108078335731074
print(cross_entropy_loss(np.array([y1,y2,y3,y4]),np.array([t1,t2,t3,t4]))) # 0.95876511607314


def cross_entropy_loss1(y,t):
    if y.ndim == 1:
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)

    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(np.log(y[np.arange(batch_size),t] + delta)) / batch_size

print('-----------')
print(cross_entropy_loss1(np.array(y1),np.array([2]))) # 0.510825457099338
print(cross_entropy_loss1(np.array(y2),np.array([2]))) # 2.302584092994546
print(cross_entropy_loss1(np.array([y1,y2]),np.array([2,2]))) # 1.406704775046942
print(cross_entropy_loss1(np.array([y1,y2,y3]),np.array([2,2,3]))) # 1.108078335731074
print(cross_entropy_loss1(np.array([y1,y2,y3,y4]),np.array([2,2,3,4]))) # 0.95876511607314
