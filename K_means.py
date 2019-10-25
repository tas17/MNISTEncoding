from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.models import load_model
from keras.datasets import mnist
from sklearn.cluster import KMeans
import numpy as np
from math import log
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
    # print('STARAAAAAAAAAAAAAAART')
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
        return False
    else:
        y = ytrain[inds[0]]
        for i in inds:
            if y != ytrain[i]:
                return True
        return False


def applyKMeans(N_CLUSTER):
    r = KMeans(n_clusters=N_CLUSTER).fit(X_train)
    # print(r.labels_.shape)
    # print(r.labels_)
    # print(r.cluster_centers_.shape)

    # ts = np.array([findClosestFromCenter(cluster, X_train) for cluster in r.cluster_centers_])
    ts = np.array([findSeveralClosestFromCenter(cluster, X_train, 1) for cluster in r.cluster_centers_])
    # ts = np.array([findSeveralClosestFromCenter(cluster, X_train, 5) for cluster in r.cluster_centers_])

    predictionTable = np.zeros(ts.shape[0])
    for c in ts:
        b = rejectOrNot(c, y_train)
        # print(c)
        # print(X_train[c], y_train[c])
        # print(r.predict(X_train[c[0]].reshape(1, -1))[0])

        # print(predictionTable)
        if b:
            # print('loop1')
            predictionTable[r.predict(X_train[c[0]].reshape(1, -1))[0]] = -1
        else:
            # print('loop2')
            predictionTable[r.predict(X_train[c[0]].reshape(1, -1))[0]] = y_train[c[0]]
        # print(y_train[c[0]])
    # print(predictionTable)
    # print(r.predict(X_train[206].reshape(1, -1)), y_train[206])
    # print(r.predict(X_train[102].reshape(1, -1)), y_train[102])
    # print(r.predict(X_train[25].reshape(1, -1)), y_train[25])
    # print(r.predict(X_train[98].reshape(1, -1)), y_train[98])
    # print(r.predict(X_train[292].reshape(1, -1)), y_train[292])
    # print(r.predict(X_train[284].reshape(1, -1)), y_train[284])
    # print(r.predict(X_train[126].reshape(1, -1)), y_train[126])
    # print(r.predict(X_train[103].reshape(1, -1)), y_train[103])
    # print(r.predict(X_train[144].reshape(1, -1)), y_train[144])
    # print(r.predict(X_train[162].reshape(1, -1)), y_train[162])

    # for i in ts:
    #     print(r.predict(X_train[i].reshape(1, -1)), y_train[i])

    # predictions = [associatePredictionManuallySelected(r, x) for x in X_test]
    # predictions = [associatePredictionSelectedByCentroid(r, x) for x in X_test]
    # print(predictionTable)
    predictions = [predictionFromTable(predictionTable, r, x) for x in X_test]
    # print(predictions)

    results = np.array([(1 if a == b else 0) for (a, b) in zip(predictions, y_test)])
    print('succeed', sum(results) / results.shape[0])
    rejects = []
    for a in predictions:
        if a+1 == 0:
            rejects.append(1)
        else:
            rejects.append(0)
    rejects = np.array(rejects)
    # print(rejects)
    print('rejected', 0 if len(rejects) <= 1 else sum(rejects) / results.shape[0])
    return sum(results) / results.shape[0], (0 if len(rejects) <= 1 else sum(rejects) / results.shape[0])

    # imgs = []
    # pred = []
    # cpt = 0
    # for i in range(100):
    #     if results[i] == 0:
    #         imgs.append(x_test[i])
    #         pred.append(associatePredictionSelectedByCentroid(r, X_test[i]))
    #         cpt += 1
    #     if cpt >= 10:
    #         break
    # imgs = np.array(imgs)
    # imgs = imgs.reshape((imgs.shape[0]*image_size, image_size))
    # imgs = (imgs * 255).astype(np.uint8)
    #
    # histo = np.zeros(10)
    # revHisto = np.zeros(10)
    # for i, r in enumerate(results):
    #     if r == 1:
    #         histo[y_test[i]] += 1
    #     else:
    #         revHisto[y_test[i]] += 1
    # s = histo + revHisto
    # print("Count of succeeds", (histo/s).round(2))
    # print("Count of failures", (revHisto/s).round(2))
    #
    # print(pred)
    # plt.figure()
    # plt.axis('off')
    # plt.title('First wrong predictions')
    # plt.imshow(imgs, interpolation='none', cmap='gray')
    # plt.show()


def entropy(d):
    e = 0
    s = 0
    for i in range(10):
        s += d[i]
    for i in range(10):
        if d[i] != 0:
            e -= d[i]/s*log(d[i]/s, 2)
    return e


def calculateEntropy(N_CLUSTER):
    r = KMeans(n_clusters=N_CLUSTER).fit(X_train)

    ts = np.array([findSeveralClosestFromCenter(cluster, X_train, 1) for cluster in r.cluster_centers_])

    predictionTable = []
    for i in range(ts.shape[0]):
        predictionTable.append({0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0})
    predictionTable = np.array(predictionTable)

    for i, x in enumerate(X_train):
        lab = r.predict(x.reshape(1, -1))
        predictionTable[lab[0]][y_train[i]] += 1
    entropies = [entropy(x) for x in predictionTable]

    print(entropies)




recap = []
recap2 = []
for i in range(1):
    t = applyKMeans(500)
    recap.append(t[0])
    recap2.append(t[1])
# calculateEntropy(50)
    # recap.append(applyKMeans((1+i)*200))

print('succeed', recap)
print('rejected', recap2)

