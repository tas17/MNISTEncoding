from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.layers import Dense
from keras.models import load_model, Sequential
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import numpy as np

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

num_classes = 10
epochs = 20

layer_filters = [32, 64]

encoder = load_model("encoder16.h5")
encoder.summary()

decoder = load_model("decoder16.h5")
decoder.summary()

autoencoder = load_model("autoEncoder16.h5")
autoencoder.summary()

# ind1 = np.array([i for i, j in enumerate(y_train) if (j == 1)])[0]
# x_test1 = x_train[ind1]
# latent1 = encoder.predict(x_test1)
# ind2 = np.array([i for i, j in enumerate(y_train) if (j == 2)])[0]
# x_test2 = x_train[ind2]
# latent2 = encoder.predict(x_test2)
# ind3 = np.array([i for i, j in enumerate(y_train) if (j == 3)])[0]
# x_test3 = x_train[ind3]
# latent3 = encoder.predict(x_test3)
# ind4 = np.array([i for i, j in enumerate(y_train) if (j == 4)])[0]
# x_test4 = x_train[ind4]
# latent4 = encoder.predict(x_test4)
# ind5 = np.array([i for i, j in enumerate(y_train) if (j == 5)])[0]
# x_test5 = x_train[ind5]
# latent5 = encoder.predict(x_test5)
# ind6 = np.array([i for i, j in enumerate(y_train) if (j == 6)])[0]
# x_test6 = x_train[ind6]
# latent6 = encoder.predict(x_test6)
# ind7 = np.array([i for i, j in enumerate(y_train) if (j == 7)])[0]
# x_test7 = x_train[ind7]
# latent7 = encoder.predict(x_test7)
# ind8 = np.array([i for i, j in enumerate(y_train) if (j == 8)])[0]
# x_test8 = x_train[ind8]
# latent8 = encoder.predict(x_test8)
# ind9 = np.array([i for i, j in enumerate(y_train) if (j == 9)])[0]
# x_test9 = x_train[ind9]
# latent9 = encoder.predict(x_test9)
# ind0 = np.array([i for i, j in enumerate(y_train) if (j == 0)])[0]
# x_test0 = x_train[ind0]
# latent0 = encoder.predict(x_test0)
#
# latents = np.array([latent0, latent1, latent2, latent3, latent4, latent5, latent6, latent7, latent8, latent9])

# x_decoded = autoencoder.predict(x_test)
X_test = encoder.predict(x_test)
X_train = encoder.predict(x_train)

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(latent_dim,)))
model.add(Dense(256, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    verbose=1,
                    validation_data=(X_test, y_test),
                    epochs=20)
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('MLPLatent16.h5')
