import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from bssa import BSSA
from update_strategy import UPDATE_STRATEGIES as us

import matplotlib.pyplot as plt

dataset = datasets.load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size = 0.2, random_state=42)
problem_dim = dataset.data.shape[1]
pop_size = 40
ub = 1
lb = 0
tf = lambda x:1/(1+np.exp(-x))
random_vector = np.random.rand(30)
bssa= BSSA(pop_size, problem_dim, tf, ub, lb, us["TCSSA2"]["S2"])
cost, food = bssa.train(100, x_train, y_train, x_test, y_test)
plt.plot(cost, label='cost')
plt.legend()
plt.show()