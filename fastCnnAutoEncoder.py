from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras.models import load_model
from keras import backend as K
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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


ind1 = np.array([i for i, j in enumerate(y_test) if (j == 1)])
x_test1 = x_test[ind1]
latent1 = encoder.predict(x_test1)

ind2 = np.array([i for i, j in enumerate(y_test) if (j == 2)])
x_test2 = x_test[ind2]
latent2 = encoder.predict(x_test2)

ind3 = np.array([i for i, j in enumerate(y_test) if (j == 3)])
x_test3 = x_test[ind3]
latent3 = encoder.predict(x_test3)

ind4 = np.array([i for i, j in enumerate(y_test) if (j == 4)])
x_test4 = x_test[ind4]
latent4 = encoder.predict(x_test4)

ind5 = np.array([i for i, j in enumerate(y_test) if (j == 5)])
x_test5 = x_test[ind5]
latent5 = encoder.predict(x_test5)

ind6 = np.array([i for i, j in enumerate(y_test) if (j == 6)])
x_test6 = x_test[ind6]
latent6 = encoder.predict(x_test6)

ind7 = np.array([i for i, j in enumerate(y_test) if (j == 7)])
x_test7 = x_test[ind7]
latent7 = encoder.predict(x_test7)

ind8 = np.array([i for i, j in enumerate(y_test) if (j == 8)])
x_test8 = x_test[ind8]
latent8 = encoder.predict(x_test8)

ind9 = np.array([i for i, j in enumerate(y_test) if (j == 9)])
x_test9 = x_test[ind9]
latent9 = encoder.predict(x_test9)

ind0 = np.array([i for i, j in enumerate(y_test) if (j == 0)])
x_test0 = x_test[ind0]
latent0 = encoder.predict(x_test0)


latent = encoder.predict(x_test)
print(latent.shape)
new = decoder.predict(latent)
print(new.shape)
x_decoded = new


rows, cols = 10, 30
num = rows * cols
imgs = np.concatenate([x_test[:num], x_decoded[:num]])
imgs = imgs.reshape((rows * 2, cols, image_size, image_size))
imgs = np.vstack(np.split(imgs, rows, axis=1))
imgs = imgs.reshape((rows * 2, -1, image_size, image_size))
imgs = np.vstack([np.hstack(i) for i in imgs])
imgs = (imgs * 255).astype(np.uint8)
plt.figure()
plt.axis('off')
plt.title('Input is above, output is under')
plt.imshow(imgs, interpolation='none', cmap='gray')
Image.fromarray(imgs).save('result.png')
plt.show()

rep_x = np.append(latent1, latent2, axis=0)
rep_x = np.append(rep_x, latent3, axis=0)
rep_x = np.append(rep_x, latent4, axis=0)
rep_x = np.append(rep_x, latent5, axis=0)
rep_x = np.append(rep_x, latent6, axis=0)
rep_x = np.append(rep_x, latent7, axis=0)
rep_x = np.append(rep_x, latent8, axis=0)
rep_x = np.append(rep_x, latent9, axis=0)
rep_x = np.append(rep_x, latent0, axis=0)


from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2).fit_transform(rep_x)
print(X_embedded)
print(X_embedded.shape)
print(X_embedded[:, 0])
print(X_embedded[:, 1])
a0 = latent1.shape[0]
a1 = latent1.shape[0]+latent2.shape[0]
a2 = a1 + latent3.shape[0]
a3 = a2 + latent4.shape[0]
a4 = a3 + latent5.shape[0]
a5 = a4 + latent6.shape[0]
a6 = a5 + latent7.shape[0]
a7 = a6 + latent8.shape[0]
a8 = a7 + latent9.shape[0]
a9 = a8 + latent0.shape[0]
plt.plot(X_embedded[:a0, 0], X_embedded[:a0, 1], color='b', linestyle="",marker="o")
plt.plot(X_embedded[a0:a1, 0], X_embedded[a0:a1, 1], color='g', linestyle="",marker="o")
plt.plot(X_embedded[a1:a2, 0], X_embedded[a1:a2, 1], color='r', linestyle="",marker="o")
plt.plot(X_embedded[a2:a3, 0], X_embedded[a2:a3, 1], color='c', linestyle="",marker="o")
plt.plot(X_embedded[a3:a4, 0], X_embedded[a3:a4, 1], color='m', linestyle="",marker="o")
plt.plot(X_embedded[a4:a5, 0], X_embedded[a4:a5, 1], color='y', linestyle="",marker="o")
plt.plot(X_embedded[a5:a6, 0], X_embedded[a5:a6, 1], color='k', linestyle="",marker="o")
plt.plot(X_embedded[a6:a7, 0], X_embedded[a6:a7, 1], color='0.75', linestyle="",marker="o")
plt.plot(X_embedded[a7:a8, 0], X_embedded[a7:a8, 1], color='0.5', linestyle="",marker="o")
plt.plot(X_embedded[a8:a9, 0], X_embedded[a8:a9, 1], color='0.25', linestyle="",marker="o")
plt.show()


