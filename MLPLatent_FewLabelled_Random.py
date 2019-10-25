from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.layers import Dense
from keras.models import load_model, Sequential
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np

np.random.seed(1337)


def main(nb_labelled):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    image_size = x_train.shape[1]
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    latent_dim = 16
    num_classes = 10
    epochs = 20

    encoder = load_model("encoder16.h5")

    decoder = load_model("decoder16.h5")

    autoencoder = load_model("autoEncoder16.h5")

    X_train = []
    Y_train = []
    for k in range(10):
        ind = np.array([i for i, j in enumerate(y_train) if (j == k)])[:nb_labelled]
        for i in ind:
            X_train.append(encoder.predict(x_train[i].reshape([-1, image_size, image_size, 1])))
            Y_train.append(k)
    X_train = np.array(X_train).reshape(nb_labelled*10, latent_dim)
    Y_train = np.array(Y_train)

    # latents = np.concatenate((latent1, latent2, latent3, latent4, latent5, latent6, latent7, latent8, latent9, latent0))

    X_test = encoder.predict(x_test)
    # X_train = encoder.predict(x_train)

    Y_train = to_categorical(Y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(latent_dim,)))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    history = model.fit(X_train, Y_train,
                        batch_size=1,
                        verbose=0,
                        epochs=epochs,
                        validation_data=(X_test, y_test))
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Nb Labelled :', str(nb_labelled))
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return score[1]


res = []
for i in range(50):
    res.append(main(i+1))

print(res)
