from salp import Salp
from accuracy import cal_cost
import numpy as np
class BSSA:
    
    def __init__(self, pop, dim, tf, ub, lb, upate_strategy):
        self.__pop = [Salp(dim, ub, lb) for i in range(pop)]
        self.__tf = tf
        self.__ub = ub
        self.__lb = lb
        self.__food_history = []
        self.__cost_history = []
        self.__update_strategy = upate_strategy
        self.__cost_func = cal_cost

    def train(self, max_iteration, train_data, train_target, test_data, test_target):
        self.__food_history = []
        self.__cost_history = []
        us = self.__update_strategy
        tf = self.__tf
        cf = self.__cost_func
        #measure accuracy and sort respect to the fitness
        for s in self.__pop:
            c = cf(s.get_position(), train_data, train_target)
            s.set_cost(c)

        self.__pop = sorted(self.__pop, key=lambda x:x.get_cost())
        #select the best one as food
        food = self.__pop[0]
        #it should be sorted
        for i in range(max_iteration):
            c1 = us(i, max_iteration)
            for this_s, pre_s, j in zip(self.__pop, 
                                        self.__pop[-1:] + self.__pop[:-1], 
                                        range(len(self.__pop))):
                if j<= len(self.__pop)/2.0:
                    c2 = np.random.rand(this_s.get_dim())
                    c3 = np.random.rand(this_s.get_dim())
                    c3 = np.array(list(map(lambda x:-1 if x<0.5 else 1, c3)))
                    this_s.set_position(food.get_position() + c1*c2*c3)
                else:
                    this_s.set_position(this_s.get_position() + pre_s.get_position())

            for s in self.__pop:
                u = self.__ub
                l = self.__lb
                pos = s.get_position()
                pos = np.array(list(map(lambda x:l if x<l else(u if x>u else x), pos)))
                s.set_position(pos)
                c = cf(s.get_position(), train_data, train_target)
                if c < food.get_cost():
                    food = s
            self.__food_history.append(food)      
            self.__cost_history.append(food.get_cost())
            print("iteratation {} cost {}".format(i, food.get_cost()))
        return self.__cost_history, self.__food_history