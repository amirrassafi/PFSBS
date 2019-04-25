import numpy as np
import pandas as pd


def load_dataset(path, label_col):
    x = []
    y = []
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            l = line.split(',')
            data = [None if '?' in a else float(a) for a in l]
            y.append(int(data[label_col]))
            del(data[label_col])
            x.append(data)
    return np.array(x), np.array(y)

def replace_none_with_zero(data):
    data = np.array(data)
    data[data == None] = 0
    return data

def load_hepatitis():
    x, y = load_dataset("datasets/hepatit/hepatitis.data", 0)
    x = replace_none_with_zero(x)
    return x, y

def load_mice():
    mice = pd.read_excel("datasets/miceprotein/data.xls")
    keys = ["class", "Genotype", "Treatment", "Behavior"]
    for key in keys:
       mice[key] =  pd.factorize(mice[key])[0]
    y = mice["class"]
    mice = mice.drop(columns=["MouseID", "class"])
    mice = mice.fillna(mice.mean())
    x = np.array(mice.values)
    y = np.array(y.values)
    return x, y
             
def load_epileptic():
    epileptic = pd.read_csv("datasets/epileptic/data.csv")
    y = epileptic["y"].values
    epileptic = epileptic.drop(columns=["y","Unnamed: 0"])
    x = epileptic.values
    return x, y

if __name__ == "__main__":
    #x, y = load_dataset("datasets/hepatit/hepatitis.data", 2)
    #print(x)
    #print(replace_none_with_zero(x))
    x, y = load_epileptic()
    print(x.shape)
    
            
