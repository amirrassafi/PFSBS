
from sklearn import datasets
from sklearn.model_selection import train_test_split
from accuracy import cal_cost_knn, cal_cost_svm, cal_cost_tree
from accuracy import test_acc_knn, test_acc_svm, test_acc_tree
from sklearn import preprocessing
import numpy as np
from dataset import load_dataset, replace_none_with_zero
from dataset import load_mice, load_epileptic


#breast cancer     
#dataset = datasets.load_breast_cancer()
#data_x = dataset.data
#data_y = dataset.target
#print(data_x[0], data_x.shape)

# hepatitis
#data_x, data_y = load_dataset("datasets/hepatit/hepatitis.data", 0)
#data_x = replace_none_with_zero(data_x)
#data_x = preprocessing.scale(data_x)

#Diabetes
#dataset = datasets.load_diabetes()
#dataset.data = preprocessing.scale(dataset.data)
#data_x = dataset.data
#data_y = dataset.target

#mice
#x, y = load_mice()
#x = preprocessing.scale(x)


#epileptic
x, y = load_epileptic()
x = preprocessing.scale(x)
print(x)
data_x = np.array(x)
data_y = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.25, random_state = 0)
problem_dim = len(x_train[0])

#dim = np.round(np.random.rand((problem_dim)))
dim = np.ones(problem_dim)
#dim[0:15] = 0
print(dim, sum(dim))

#KNN
_, e, _= cal_cost_knn(dim, np.copy(x_train), np.copy(y_train))
e = 1 - e
print("KNN train accuracy = {}".format(e))
a = test_acc_knn(dim, x_test, y_test, x_train, y_train) 
print("KNN test accuracy = {}".format(a))

#SVM
_, e, _= cal_cost_svm(dim, np.array(x_train), np.array(y_train))
e = 1 - e
print("SVM(SVC) train accuracy = {}".format(e))
a = test_acc_svm(dim, x_test, y_test, x_train, y_train) 
print("SVM(SVC) test accuracy = {}".format(a))

#TREE
_, e, _= cal_cost_svm(dim, np.array(x_train), np.array(y_train))
e = 1 - e
print("Tree train accuracy = {}".format(e))
a = test_acc_tree(dim, x_test, y_test, x_train, y_train) 
print("Tree test accuracy = {}".format(a))


