
from sklearn import datasets
from sklearn.model_selection import train_test_split
from accuracy import cal_cost_knn, cal_cost_svm, cal_cost_tree
from accuracy import test_acc_knn, test_acc_svm, test_acc_tree

import numpy as np



# with open("dataset/waveform.data") as f:
#     lines = f.readlines()
#     data_x = []
#     data_y = []
#     for line in lines:
#         l = [float(d) for d in line.split(",")]
#         data_x.append(l[: -1])
#         data_y.append(l[-1])
        
dataset = datasets.load_breast_cancer()
data_x = dataset.data
data_y = dataset.target
# print(data_x)
# print(data_y)

data_x = np.array(data_x)
data_y = np.array(data_y)
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.25)
problem_dim = len(x_train[0])

dim = np.round(np.random.rand((problem_dim)))
print(dim, sum(dim))
#KNN
_, e, _= cal_cost_knn(dim, x_train, y_train)
e = 1 - e
print("KNN train accuracy = {}".format(e))
a = test_acc_knn(dim, x_test, y_test, x_train, y_train) 
print("KNN test accuracy = {}".format(a))

#SVM
_, e, _= cal_cost_svm(dim, x_train, y_train)
e = 1 - e
print("SVM(SVC) train accuracy = {}".format(e))
a = test_acc_svm(dim, x_test, y_test, x_train, y_train) 
print("SVM(SVC) test accuracy = {}".format(a))

#TREE
_, e, _= cal_cost_svm(dim, x_train, y_train)
e = 1 - e
print("Tree train accuracy = {}".format(e))
a = test_acc_tree(dim, x_test, y_test, x_train, y_train) 
print("Tree test accuracy = {}".format(a))


