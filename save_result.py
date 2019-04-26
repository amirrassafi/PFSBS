import pickle
import os 
import numpy as np 


#use 1, 2, 3, 4 respect to persian paper for saving data
def save_history(dataset_no, data, name):
    with open('result', mode='rb') as res:
        try:
            d = pickle.load(res)
        except Exception as e:
            print(e)
            d = {}
    if str(dataset_no) in d.keys():
        
        if name in d[str(dataset_no)].keys():
            d[str(dataset_no)][name].append(data)
        else:
            d[str(dataset_no)][name] =  [data]
    else:
        d[str(dataset_no)] = {name:  [data]}
    with open('result', mode='wb') as res:
        pickle.dump(d, res)

def load_history(dataset_no, name):
    with open("result", mode='rb') as res:
        d = pickle.load(res)
    data = d[str(dataset_no)][name]
    return data

def save_cost_history(dataset_no, data):
    save_history(dataset_no, data, "cost")

def save_acc_history(dataset_no, data):
    save_history(dataset_no, data, "acc")

def save_sf_history(dataset_no, data):
    save_history(dataset_no, data, "sf")



def load_cost_history(dataset_no):
    return load_history(dataset_no, "cost")


def load_acc_history(dataset_no):
    return load_history(dataset_no, "acc")

def load_sf_history(dataset_no):
    return load_history(dataset_no, "sf")

def save_breast_cancer_cost(data):
    save_cost_history(1, data)

def load_breast_cancer_cost():
    return load_cost_history(1)

def save_hepatitis_cost(data):
    save_cost_history(2, data)

def load_hepatitis_cost():
    return load_cost_history(2)

def save_diabeties_cost(data):
    save_cost_history(3, data)

def load_diabeties_cost():
    return load_cost_history(3)

def save_mice_cost(data):
    save_cost_history(4, data)

def load_mice_cost():
    return load_cost_history(4)


def save_breast_cancer_acc(data):
    save_acc_history(1, data)

def load_breast_cancer_acc():
    return load_acc_history(1)

def save_hepatitis_acc(data):
    save_acc_history(2, data)

def load_hepatitis_acc():
    return load_acc_history(2)

def save_diabeties_acc(data):
    save_acc_history(3, data)

def load_diabeties_acc():
    return load_acc_history(3)

def save_mice_acc(data):
    save_acc_history(4, data)

def load_mice_acc():
    return load_acc_history(4)

def save_breast_cancer_sf(data):
    save_sf_history(1, data)

def load_breast_cancer_sf():
    return load_sf_history(1)

def save_hepatitis_sf(data):
    save_sf_history(2, data)

def load_hepatitis_sf():
    return load_sf_history(2)

def save_diabeties_sf(data):
    save_sf_history(3, data)

def load_diabeties_sf():
    return load_sf_history(3)

def save_mice_sf(data):
    save_sf_history(4, data)

def load_mice_sf():
    return load_sf_history(4)

if __name__ == "__main__":
    exit()
    save_mice_cost([1, 2, 3])
    save_mice_acc([2, 4, 6])
    save_mice_sf([3, 5, 9])
    print(load_mice_cost())
    print(load_mice_acc())
    print(load_mice_sf())
