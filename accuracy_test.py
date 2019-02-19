
from sklearn import datasets
from sklearn.model_selection import train_test_split
from accuracy import cal_cost_knn, cal_cost_svm
import numpy as np


dataset = datasets.load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size = 0.2, random_state=42)

problem_dim = dataset.data.shape[1]

_, e, _= cal_cost_knn(np.ones(problem_dim), x_train, y_train) 
e = 1 - e 
print("KNN train accuracy = {}".format(e))

_, e, _= cal_cost_svm(np.ones(problem_dim), x_train, y_train)  
e = 1 - e
print("SVM(SVC) train accuracy = {}".format(e))

