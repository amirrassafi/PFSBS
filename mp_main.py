import multiprocessing as mp
import time

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from bssa import BSSA
from update_strategy import UPDATE_STRATEGIES as us

import matplotlib.pyplot as plt
import main
import logging

logger = logging.getLogger("main.mp__main")

#Load dataset
dataset = datasets.load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size = 0.2, random_state=42)
#Set and define parameter
problem_dim = dataset.data.shape[1]
pop_size = 30
ub = 1
lb = 0
tf = lambda x:1/(1+np.exp(-x))
strategies = ["TCSSA1", "TCSSA1", "TCSSA1", "TCSSA1"]
sub_chains = ["S1", "S2", "S3", "S4"]

#Define bssa object
bssa_list = []
for st, su in zip(strategies, sub_chains):
    bssa_list.append(BSSA(pop_size, problem_dim, tf, ub, lb, us[st][su]))

iter
#Train
t = time.time()
for i in range(10):
    p_l = []
    for bssa in bssa_list:
        p = mp.Process(target = bssa.train, args=(2, x_train, y_train, x_test, y_test, ))
        p.start()
        p_l.append(p)
    for p in p_l:
        p.join()
    #bssa_list = sorted(bssa_list, key= lambda x: x.get_best_cost())
    print([a.get_best_cost() for a in bssa_list])
    #logger.info("General Iteratation {} best_cost {}".format(i, bssa_list[0].get_best_cost()))
    #for bssa in bssa_list[1:]:
    #    bssa.replace_with_worst_salp(bssa_list[0].get_best_salp())
    #logger.info("The best salp of each bssa replaced with worst one")

logger.info("Time = {}".format(time.time() - t))