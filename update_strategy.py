import numpy as np
import matplotlib.pyplot as plt

UPDATE_STRATEGIES ={
    "TCSSA1":
            {
                "S1":(lambda t, T: -2.05 * t / T + 2.55),
                "S2":(lambda t, T: -2.05 * t / T + 2.55),
                "S3":(lambda t, T: (-2.0 * t**3) / T**3 + 2.5),
                "S4":(lambda t, T: (-2.0 * t**3) / T**3 + 2.5)
            },
    "TCSSA2":
            {
                "S1":(lambda t, T: 2.5 + 2.0*(t*1.0/T)**2 - 4.0*t/T),
                "S2":(lambda t, T: 0.5 + 2.0*np.exp(-(4.0*t/T)**2)),
                "S3":(lambda t, T: (-2.0*t**3/T**3) + 2.5),
                "S4":(lambda t, T: 2.5 - (2.0*np.log(t)/np.log(T))) 
            },
    "TCSSA3":
            {
                "S1":(lambda t, T: 1.95 - 2.0*t**(1.0/3)*T**(1.0/3)),
                "S2":(lambda t, T: 1.95 - 2.0*t**(1.0/3)*T**(1.0/3)),
                "S3":(lambda t, T: -2.0 * t**3/T**3 + 2.5),
                "S4":(lambda t, T: -2.0 * t**3/T**3 + 2.5)  
            }
}

if __name__ == '__main__':
    T = 200
    fig = plt.figure()
    l = 1
    for st in UPDATE_STRATEGIES:
        for sb in UPDATE_STRATEGIES[st]:
            f = UPDATE_STRATEGIES[st][sb]
            s = [f(t, T) for t in range(T)]
            ax = fig.add_subplot(3, 4, l)
            l = l+1
            ax.plot(s, label=st+"-"+sb)
            ax.legend()
    plt.show()
