from salp import Salp

class BSSA:
    def __init__(self, pop, dim, tf, ub, lb):
        self.__pop_size = [Salp(dim, ub, lb) for i in range(pop)]
        self.__tf = tf
        self.__ub = ub
        self.__lb = lb
        self.__fitness_list = []

    def train(self, max_iteration, train_data, test_data):
        #measure accuracy and sort respect to the fitness
        for i in range(max_iteration):
            
            pass