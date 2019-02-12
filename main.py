import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from bssa import BSSA
from update_strategy import UPDATE_STRATEGIES as us

import matplotlib.pyplot as plt
import logging

#Config logger
logger = logging.getLogger("main")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
fh = logging.FileHandler("log.txt")
ch.setLevel(logging.INFO)
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)
#Load dataset
dataset = datasets.load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size = 0.2, random_state=42)
#Set and define parameter
problem_dim = dataset.data.shape[1]
pop_size = 30
ub = 1
lb = 0
tf = lambda x:1/(1+np.exp(-x))
strategy = "TCSSA1"
sub_chain = "S1"
#Define bssa object
logger.info("-" * 20 + "New Run" + "-" * 20+"pop_size = {} problem_dim = {} Method = {}".format(pop_size, problem_dim, strategy+" "+sub_chain))
bssa= BSSA(pop_size, problem_dim, tf, ub, lb, us[strategy][sub_chain])
cost, food = bssa.train(100, x_train, y_train, x_test, y_test)
logger.info("best is {}, number of selected feature is {}".format(bssa.get_best_selected(), sum(bssa.get_best_selected())))
#Plot cost
cost_fig = plt.figure()
cost_ax = cost_fig.add_subplot(1, 1, 1)
cost_ax.plot(cost, label='cost')
cost_ax.legend()
food_fig = plt.figure()
food_ax = food_fig.add_subplot(1, 1, 1)
food_ax.plot(food, label = "num of selected feature")
food_ax.legend()
plt.show()