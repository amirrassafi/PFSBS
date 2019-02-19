import numpy as np
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.model_selection import cross_val_score as cv
from sklearn.metrics import accuracy_score as ac_sc

alpha = 0.01

def cal_cost(x, trn, trg):
    x = [int(a) for a in np.round(x)]
    x_index = [i for i in range(len(x)) if x[i]==1]
    if sum(x) == 0 : return np.inf
    trn = trn.reshape(trn.shape[1], -1)
    trn = trn[x_index, :]
    trn = np.transpose(trn)
    clf = knn(n_neighbors=5)
    clf.fit(trn, trg)
    score = cv(clf, trn, trg, cv=5, scoring="accuracy")
    score = np.average(score)
    error = 1 - score
    return (1-alpha)*error + alpha * (sum(x)/len(x)), error, sum(x)*1.0/len(x)

def test_acc(x, tst, tst_trg, trn, trn_trg):
    x = [int(a) for a in np.round(x)]
    x = [i for i in range(len(x)) if x[i]==1]
    tst = tst.reshape(tst.shape[1], -1)
    tst = tst[x, :]
    tst = np.transpose(tst)
    trn = trn.reshape(trn.shape[1], -1)
    trn = trn[x, :]
    trn = np.transpose(trn)
    clf = knn(n_neighbors=5)
    clf.fit(trn, trn_trg)
    tst_pred = clf.predict(tst)
    acc = ac_sc(tst_trg, tst_pred)
    print(acc)
    return acc


    