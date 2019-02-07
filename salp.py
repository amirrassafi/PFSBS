import numpy as np
class Salp:
    def __init__(self, dim, ub, lb):
        self.__pos = np.random.uniform(low=lb, high=ub, size=dim)
        self.__pos_history = []
        self.__cost_history = []
        self.__cost = np.inf
        self.__dim = dim
        
    def get_dim(self):
        return self.__dim

    def set_cost(self, cost):
        self.__cost_history.append(self.__cost)
        self.__cost= cost

    def get_cost(self):
        return self.__cost

    def get_position(self):
        return self.__pos

    def set_position(self, pos):
        pos = np.array(pos)
        if pos.shape == self.__pos.shape:
            self.__pos_history.append(self.__pos)
            self.__pos = pos
        else:
            raise Exception("Given dimension isn't same with salp own dimension ")
            
    def get_position_history(self):
        return self.__pos_history

    def reset_position_history(self):
        self.__pos_history = []   


if __name__=='__main__':
    s1 = Salp(3, 1, 0)
    s2 = Salp(3, 1, 0) 
    print(s1.get_position())
    s1.set_position([1, 2, 3])
    print("Id s1 is {}".format(id(s1)))
    print("Id s2 is {}".format(id(s2)))