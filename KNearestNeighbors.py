from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.models import load_model
from keras.datasets import mnist
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1337)


def distance(a, b):
    return np.linalg.norm(a-b)


def findClosestFromCenter(cluster, X_train):
    t = np.array([distance(cluster, xi) for xi in X_train])
    dmin = t[0]
    index = 0
    for i, d in enumerate(t):
        if d < dmin:
            dmin = d
            index = i
    return index


def findSeveralClosestFromCenter(cluster, X_train, N):
    t = np.array([distance(cluster, xi) for xi in X_train])
    dmax = t[0]
    index = 0
    ds = [dmax]
    inds = [index]
    for i, d in enumerate(t):
        if len(ds) < N:
            # print('loop1', i, d)
            ds.append(d)
            inds.append(i)
            dmax = max(d, dmax)
        else:
            # print('loop2', i, d, dmax)
            ds = np.array(ds)
            inds = np.array(inds)
            if d < dmax:
                for j, e in enumerate(ds):
                    # print('subloop', j, e, dmax)
                    if e == dmax:
                        ds[j] = d
                        inds[j] = i
                        dmax = np.amax(ds)
                        # print('subsubloop', dmax, inds)
                        break
    return inds


def associatePredictionManuallySelected(r, x):
    lab = r.predict(x.reshape(1, -1))
    if lab[0] == 0:
        return 0
    if lab[0] == 8:
        return 1
    if lab[0] == 1:
        return 2
    if lab[0] == 4:
        return 3
    if lab[0] == 5:
        return 4
    if lab[0] == 6:
        return 5
    if lab[0] == 9:
        return 6
    if lab[0] == 3:
        return 7
    if lab[0] == 7:
        return 8
    if lab[0] == 2:
        return 9


def associatePredictionSelectedByCentroid(r, x):
    lab = r.predict(x.reshape(1, -1))
    if lab[0] == 0:
        return 0
    if lab[0] == 1:
        return 2
    if lab[0] == 2:
        return 9
    if lab[0] == 3:
        return 7
    if lab[0] == 4:
        return 3
    if lab[0] == 5:
        return 1
    if lab[0] == 6:
        return 8
    if lab[0] == 7:
        return 8
    if lab[0] == 8:
        return 1
    if lab[0] == 9:
        return 6


def predictionFromTable(table, r, x):
    lab = r.predict(x.reshape(1, -1))
    return table[lab[0]]


(x_train, y_train), (x_test, y_test) = mnist.load_data()

image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

encoder = load_model("encoder16.h5")

X_test = encoder.predict(x_test)
X_train = encoder.predict(x_train)


def rejectOrNot(inds, ytrain):
    if len(inds) == 1:
        return 0
    else:
        y = ytrain[inds[0]]
        for i in inds:
            if y != ytrain[i]:
                return 1
        return 0


def applyKNearNeighbor(N_NEIGHBOR):
    ts = [findSeveralClosestFromCenter(xtest, X_train, N_NEIGHBOR) for xtest in X_test]
    rejects = [rejectOrNot(t, y_train) for t in ts]
    results = np.array([1 if (a == y_train[b[0]] and c == 0) else 0 for a, b, c in zip(y_test, ts, rejects)])

    print('results', sum(results) / 10000)
    print('rejects', sum(rejects) / 10000)
    return sum(results) / 10000, sum(rejects) / 10000


recap = []
recap2 = []
# for i in range(10):
t = applyKNearNeighbor(3)
recap.append(t[0])
recap2.append(t[1])
t = applyKNearNeighbor(5)
recap.append(t[0])
recap2.append(t[1])
    # recap.append(applyKMeans((1+i)*200))

print('succeed', recap)
print('rejected', recap2)

