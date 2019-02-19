import numpy as np
import matplotlib.pyplot as plt

def T1_S1(t, T):
    return -2.05 * t / T + 2.55

def T1_S2(t, T):
    return T1_S1(t, T)

def T1_S3(t, T):
    return (-2.0 * t**3) / T**3 + 2.5

def T1_S4(t, T):
    return T1_S3(t, T)

def T2_S1(t, T):
    return 2.5 + 2.0*(t*1.0/T)**2 - 4.0*t/T

def T2_S2(t, T):
    return 0.5 + 2.0*np.exp(-(4.0*t/T)**2)

def T2_S3(t, T):
    return (-2.0*t**3/T**3) + 2.5

def T2_S4(t, T):
    return 2.5 - (2.0*np.log(t)/np.log(T))

def T3_S1(t, T):
    return 1.95 - 2.0*t**(1.0/3)/T**(1.0/3)

def T3_S2(t, T):
    return T3_S1(t, T)

def T3_S3(t, T):
    return (-2.0 * t**3/T**3) + 2.5

def T3_S4(t, T):
    return T3_S3(t, T)

UPDATE_STRATEGIES ={
    "TCSSA1":
            {
                "S1":T1_S1,
                "S2":T1_S2,
                "S3":T1_S3,
                "S4":T1_S4
            },
    "TCSSA2":
            {
                "S1":T2_S1,
                "S2":T2_S2,
                "S3":T2_S3,
                "S4":T2_S4 
            },
    "TCSSA3":
            {
                "S1":T3_S1,
                "S2":T3_S2,
                "S3":T3_S3,
                "S4":T3_S4
            }
}

if __name__ == '__main__':
    T = 200
    fig = plt.figure()
    l = 1
    for st, c in zip(UPDATE_STRATEGIES, ['r', 'g', 'b']):
        for sb in sorted(UPDATE_STRATEGIES[st]):
            f = UPDATE_STRATEGIES[st][sb]
            s = [f(t, T) for t in range(T)]
            ax = fig.add_subplot(3, 4, l)
            l = l+1
            ax.plot(s, c, label=st+"-"+sb)
            ax.legend()
    plt.show()
