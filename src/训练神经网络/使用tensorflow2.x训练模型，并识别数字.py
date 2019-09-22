# tensorflow 2.x

import tensorflow as tf

import os
os.environ['TF_CPP_MIN)LOG_LEVEL'] = '2'
minist = tf.keras.datasets.mnist

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(50,activation='sigmoid',input_shape=(784,)),
        tf.keras.layers.Dense(100, activation='sigmoid', input_shape=(50,)),
        # 为了防止过度拟合，设置丢弃的比例（随机将输入单元设置为0），这里是20%
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10,activation='softmax')
    ])

    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model

model = create_model()

(train_images,train_labels),(test_images,test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:10000]
test_labels = test_labels[:10000]

train_images = train_images[:10000].reshape(-1,28*28) / 255.0
test_images = test_images[:10000].reshape(-1,28*28) / 255.0

model.fit(train_images,train_labels,epochs=20)

model.save('mnist.model')









