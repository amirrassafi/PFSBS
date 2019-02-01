import numpy as np
class Salp:
    def __init__(self, dim, ub, lb):
        self.__pos = np.random.uniform(low=lb, high=ub, size=dim)
        self.__pos_history = []
    
    def get_position(self):
        return self.__pos

    def set_position(self, pos):
        if pos.shape == self.__pos:
            self.__pos_history.append(self.__pos)
            self.__pos = pos
        else:
            raise Exception("Given dimension isn't same with salp own dimension ")
            
    def get_position_history(self):
        return self.__pos_history

    def reset_position_history(self):
        self.__pos_history = []   
class BSSA:
    def __init__(self, pop, dim, tf, ub, lb):
        self.__pop_size = [Salp(dim, ub, lb) for i in range(pop)]
        self.__tf = tf
        self.__ub = ub
        self.__lb = lb


    def train(self, max_iteration, train_data, test_data):
        pass
    