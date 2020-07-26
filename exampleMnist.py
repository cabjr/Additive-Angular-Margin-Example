import numpy as np
np.random.seed(1337)
from AM_Softmax import *
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense,Input,Conv2D,MaxPooling2D,Dropout,BatchNormalization,Activation,Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import regularizers
import tensorflow as tf
import os

batch_size = 300
nb_classes = 10
nb_epoch = 50
weight_decay = 1e-4



def vgg_block(x, filters, layers):
    for _ in range(layers):
        x = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    return x

mnist = tf.keras.datasets.mnist

(X, y), (X_test, y_test) = mnist.load_data()

X = X[:, :, :, np.newaxis].astype('float32') / 255
X_test = X_test[:, :, :, np.newaxis].astype('float32') / 255

y = tf.keras.utils.to_categorical(y, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)


input = Input(shape=(28, 28, 1))
y_in = Input(shape=(10,))
x = vgg_block(input, 16, 2)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = vgg_block(x, 32, 2)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = vgg_block(x, 64, 2)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Flatten()(x)
x = Dense(512, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(weight_decay))(x)
x = BatchNormalization()(x)
output = ArcNorm(10, regularizer=regularizers.l2(weight_decay))([x, y_in])
#output = Dense(10, activation='softmax')(y)
model = Model(inputs=[input,y_in], outputs=output)
model.summary()

adam = Adam()
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

history = model.fit([X,y], y,
                    batch_size=batch_size,
                    epochs=nb_epoch,
                    verbose=1,
                    validation_data=([X_test,y_test], y_test))

score = model.evaluate([X_test, y_test], y_test, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])
