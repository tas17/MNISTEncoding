import sys
from minisom import MiniSom
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

from sklearn import datasets
from sklearn.preprocessing import scale

(x_train, y_train), (x_test, y_test) = mnist.load_data()

image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
encoder = load_model("encoder16.h5")
X_test = encoder.predict(x_test)
X_train = encoder.predict(x_train)

# load the digits dataset from scikit-learn
digits = datasets.load_digits(n_class=10)
data = digits.data  # matrix where each row is a vector that represent a digit.
data = scale(data)
num = digits.target  # num[i] is the digit represented by data[i]

som = MiniSom(5, 5, 16, sigma=4,
              learning_rate=0.5, neighborhood_function='triangle')
# print(data)
# print(X_train)
som.pca_weights_init(X_train)
print("Training...")
som.train_random(X_train, 500000, verbose=True)  # random training
print("\n...ready!")

plt.figure(figsize=(8, 8))
wmap = {}
im = 0
for x, t in zip(X_train[:100], y_train[:100]):  # scatterplot
    w = som.winner(x)
    # print(w,x,t)
    wmap[w] = im
    plt.text(w[0]+.5,  w[1]+.5,  str(t),
              color=plt.cm.rainbow(t / 10.), fontdict={'weight': 'bold',  'size': 11})
    im = im + 1
plt.axis([0, som.get_weights().shape[0], 0,  som.get_weights().shape[1]])
plt.savefig('som_digts.png')
#plt.show()
predictionTable = som.labels_map(X_train[:50], y_train[:50])
# print(predictionTable)


def predictX(x, table):
    a, b = som.winner(x)
    w = (a, b)
    if not table[w]:
        x1 = predictW((a+1, b), table)
        x2 = predictW((a-1, b), table)
        x3 = predictW((a, b+1), table)
        x4 = predictW((a, b-1), table)
        # # print(x1,x2,x3,x4)
        x = None
        print("Case to treat", x1, x2, x3, x4)
        if x1 is not None:
            x = x1
        else:
            if x2 is not None:
                x = x2
            else:
                if x3 is not None:
                    x = x3
                else:
                    if x4 is not None:
                        x = x4
        if x is not None \
                and (x1 is None or x == x1) and (x2 is None or x == x2) \
                and (x3 is None or x == x3) and (x4 is None or x == x4):

            print("Returning", x)
            return x
        else:
            print("Returning None")
            return None
        # if x1 is not None:
        #     return x1
        # if x2 is not None:
        #     return x2
        # if x3 is not None:
        #     return x3
        # if x4 is not None:
        #     return x4
        return None
    else:
        elems = table[w].most_common()
        t = elems[0][0]
        p = elems[0][1]
        s = 0
        for j in elems:
            s += j[1]
        if p/s > 0.8:
            return t
        else:
            # print(elems)
            return None


def predictW(w, table):
    if not table[w]:
        return None
    else:
        return table[w].most_common()[0][0]


s = 0
rej = 0
for i, x in enumerate(X_test):
    y = predictX(X_test[i], predictionTable)
    if y is None:
        rej += 1
    else:
        if y == y_test[i]:
            s += 1
print('succeed', str(s/X_test.shape[0]))
print('rejected', str(rej/X_test.shape[0]))

