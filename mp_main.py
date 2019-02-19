import multiprocessing as mp
import time

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from bssa import BSSA
from pickable_us import UPDATE_STRATEGIES as us

import matplotlib.pyplot as plt
import main
import logging

import pickle

logger = logging.getLogger("main.mp__main")

#Load dataset
dataset = datasets.load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size = 0.2, random_state=42)
#Set and define parameter
problem_dim = dataset.data.shape[1]
pop_size = 30
ub = 1
lb = 0
def tf(x):
    return 1/(1+np.exp(-x))
strategies = ["TCSSA1", "TCSSA1", "TCSSA1", "TCSSA1"]
sub_chains = ["S1", "S2", "S3", "S4"]

#Define bssa object
bssa_list = []
for st, su in zip(strategies, sub_chains):
    bssa_list.append(BSSA(pop_size, problem_dim, tf, ub, lb, us[st][su]))
    
def train_bssa(in_object_q, out_object_q, iter_num):
    bssa = pickle.loads(in_object_q.get())
    bssa.train(iter_num, x_train, y_train, x_test, y_test)
    out_object_q.put(pickle.dumps(bssa))

iterations = 2
sync_iter = 1
#Train
t = time.time()
for i in range(iterations//sync_iter):
    p_l = []
    #Create q for send and recv object to process
    send_q_l = []
    recv_q_l = []
    for bssa in bssa_list:
        send_q_l.append(mp.Queue())
        recv_q_l.append(mp.Queue())
    #Create process
    for bssa, s, r in zip(bssa_list, send_q_l, recv_q_l):
        s.put(pickle.dumps(bssa))
        p = mp.Process(target = train_bssa, args=(s, r, sync_iter, ))
        p.start()
        p_l.append(p)
    #Wait for reading
    bssa_list = []
    for r in recv_q_l:
        bssa_list.append(pickle.loads(r.get()))
    
    bssa_list = sorted(bssa_list, key= lambda x: x.get_best_cost())
    logger.debug([a.get_best_cost() for a in bssa_list])
    for bssa in bssa_list[1:]:
        bssa.replace_with_worst_salp(bssa_list[0].get_best_salp())
    logger.info("The best salp of each bssa replaced with worst one")

logger.info("Time = {}".format(time.time() - t))

for bssa, i in zip(bssa_list, range(len(bssa_list))):
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("cost")
    ax.set_ylabel("iteration")
    ax.plot(bssa.get_cost_history(), 'g')
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("error, selected_feature history")
    ax.plot(bssa.get_error_history(), 'r', label="cross validation error")
    ax.plot(bssa.get_selected_features_history(), 'b', label="selcted features")
    ax.legend()
plt.show()