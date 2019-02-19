
from sklearn import datasets
from sklearn.model_selection import train_test_split
from accuracy import cal_cost_knn, cal_cost_svm
import numpy as np



with open("dataset/waveform.data") as f:
    lines = f.readlines()
    data_x = []
    data_y = []
    for line in lines:
        l = [float(d) for d in line.split(",")]
        data_x.append(l[: -1])
        data_y.append(l[-1])
        
#dataset = datasets.load_breast_cancer()
#data_x = dataset.data
#data_y = dataset.target
data_x = np.array(data_x)
data_y = np.array(data_y)
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state=42)
problem_dim = len(x_train[0])



_, e, _= cal_cost_knn(np.ones(problem_dim), x_train, y_train) 
e = 1 - e 
print("KNN train accuracy = {}".format(e))

_, e, _= cal_cost_svm(np.ones(problem_dim), x_train, y_train)  
e = 1 - e
print("SVM(SVC) train accuracy = {}".format(e))

