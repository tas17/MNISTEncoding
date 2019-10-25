from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.models import load_model

np.random.seed(1337)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

input_shape = (image_size, image_size, 1)
batch_size = 128
kernel_size = 3
latent_dim = 16

layer_filters = [32, 64]

encoder = load_model("encoder16.h5")
encoder.summary()

decoder = load_model("decoder16.h5")
decoder.summary()

autoencoder = load_model("autoEncoder16.h5")
autoencoder.summary()


rows, cols = 10, 30
num = rows * cols
# imgs = np.concatenate([x_train[4*cols+9], x_train[8*cols+1], x_train[2*cols+11], x_train[7*cols+28], x_train[4*cols+29],
#                        x_train[2*cols+12], x_train[4*cols+10], x_train[8*cols+12], x_train[4*cols+15], x_train[8*cols+20]])
imgs = np.concatenate([x_train[102], x_train[25], x_train[98], x_train[292], x_train[284],
                       x_train[126], x_train[103], x_train[144], x_train[162], x_train[206]])
# imgs = np.concatenate([x_train[0], x_train[1], x_train[2], x_train[3], x_train[30]])
print(imgs.shape)
imgs = imgs.reshape((10, 1, image_size, image_size))
print(imgs.shape)
imgs = imgs.reshape((10*image_size, image_size))
print(imgs.shape)
imgs = (imgs * 255).astype(np.uint8)
print(imgs.shape)

plt.figure()
plt.axis('off')
plt.title('Selected candidates for training with labels')
plt.imshow(imgs, interpolation='none', cmap='gray')
plt.show()
