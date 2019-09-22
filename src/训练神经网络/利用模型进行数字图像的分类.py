import tensorflow as tf

import os
os.environ['TF_CPP_MIN)LOG_LEVEL'] = '2'

new_model = tf.keras.models.load_model('mnist.model')
print(new_model.summary())

# 精确度

minist = tf.keras.datasets.mnist
(train_images,train_labels),(test_images,test_labels) = tf.keras.datasets.mnist.load_data()
train_labels = train_labels[:10000]
test_labels = test_labels[:10000]

train_images = train_images[:10000].reshape(-1,28*28) / 255.0
test_images = test_images[:10000].reshape(-1,28*28) / 255.0

loss,acc = new_model.evaluate(test_images,test_labels)

print("accuracy：{:5.2f}%".format(100*acc))

import numpy as np
print(np.argmax(new_model.predict(test_images[2000:2001])))
print(test_labels[2000:2001])

from PIL import Image
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()
img_show((255 * test_images[2000:2001]).reshape(28,28))