import salp
import numpy as np

pop_size = 40
ub = 1
lb = 0
problem_dim = 3
tf = lambda x:1/(1+np.exp(-x))

bssa= BSSA(pop_size, problem_dim, tf, ub, lb)
