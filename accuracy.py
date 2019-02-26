import numpy as np
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.svm import NuSVC 
from sklearn.svm import SVC 
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score as cv
from sklearn.metrics import accuracy_score as acc
from sklearn import tree 

alpha = 0.01

#Nearest neighbour for KNN
nn = 10
svm_kernel = "rbf"

def cal_cost_tree(x, trn, trg):
    x = [int(a) for a in np.round(x)]
    x_index = [i for i in range(len(x)) if x[i]==1]
    if sum(x) == 0 : return np.inf
    trn = trn.reshape(trn.shape[1], -1)
    trn = trn[x_index, :]
    trn = np.transpose(trn)
    clf = tree.DecisionTreeClassifier()
    clf.fit(trn, trg)
    pre = clf.predict(trn)
    score = acc(pre, trg)
    error = 1 - score
    return (1-alpha)*error + alpha * (sum(x)*1.0/len(x)), error, sum(x)*1.0/len(x)


def cal_cost_knn(x, trn, trg):
    x = [int(a) for a in np.round(x)]
    x_index = [i for i in range(len(x)) if x[i]==1]
    if sum(x) == 0 : return np.inf
    trn = trn.reshape(trn.shape[1], -1)
    trn = trn[x_index, :]
    trn = np.transpose(trn)
    clf = knn(n_neighbors=nn)
    clf.fit(trn, trg)
    pre = clf.predict(trn)
    score = acc(pre, trg)
    error = 1 - score
    return (1-alpha)*error + alpha * (sum(x)*1.0/len(x)), error, sum(x)*1.0/len(x)

def cal_cost_svm(x, trn, trg):
    x = [int(a) for a in np.round(x)]
    x_index = [i for i in range(len(x)) if x[i]==1]
    if sum(x) == 0 : return np.inf
    trn = trn.reshape(trn.shape[1], -1)
    trn = trn[x_index, :]
    trn = np.transpose(trn)
    clf = SVC(gamma="auto", kernel=svm_kernel)
    clf.fit(trn, trg)
    pre = clf.predict(trn)
    score = acc(pre, trg)
    error = 1 - score
    return (1-alpha)*error + alpha * (sum(x)*1.0/len(x)), error, sum(x)*1.0/len(x)


def test_acc_knn(x, tst, tst_trg, trn, trn_trg):
    x = [int(a) for a in np.round(x)]
    x = [i for i in range(len(x)) if x[i]==1]
    tst = tst.reshape(tst.shape[1], -1)
    tst = tst[x, :]
    tst = np.transpose(tst)
    trn = trn.reshape(trn.shape[1], -1)
    trn = trn[x, :]
    trn = np.transpose(trn)
    clf = knn(n_neighbors=nn)
    clf.fit(trn, trn_trg)
    tst_pred = clf.predict(tst)
    return acc(tst_trg, tst_pred)


def test_acc_svm(x, tst, tst_trg, trn, trn_trg):
    x = [int(a) for a in np.round(x)]
    x = [i for i in range(len(x)) if x[i]==1]
    tst = tst.reshape(tst.shape[1], -1)
    tst = tst[x, :]
    tst = np.transpose(tst)
    trn = trn.reshape(trn.shape[1], -1)
    trn = trn[x, :]
    trn = np.transpose(trn)
    clf = SVC(gamma="auto", kernel=svm_kernel)
    clf.fit(trn, trn_trg)
    tst_pred = clf.predict(tst)
    return acc(tst_trg, tst_pred)
    


def test_acc_tree(x, tst, tst_trg, trn, trn_trg):
    x = [int(a) for a in np.round(x)]
    x = [i for i in range(len(x)) if x[i]==1]
    tst = tst.reshape(tst.shape[1], -1)
    tst = tst[x, :]
    tst = np.transpose(tst)
    trn = trn.reshape(trn.shape[1], -1)
    trn = trn[x, :]
    trn = np.transpose(trn)
    clf = tree.DecisionTreeClassifier()
    clf.fit(trn, trn_trg)
    tst_pred = clf.predict(tst)
    return acc(tst_trg, tst_pred)
    