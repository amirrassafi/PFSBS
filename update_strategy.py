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
                "S1":(lambda t, T: 1.95 - 2.0*t**(1.0/3)/T**(1.0/3)),
                "S2":(lambda t, T: 1.95 - 2.0*t**(1.0/3)/T**(1.0/3)),
                "S3":(lambda t, T: (-2.0 * t**3/T**3) + 2.5),
                "S4":(lambda t, T: (-2.0 * t**3/T**3) + 2.5)  
            },
    "TCSSA4":
            {
                "S1":(lambda t, T: 2.0*np.sqrt(2.5 - 2.5*np.tanh(t*6/T))),
                "S2":(lambda t, T: 0.5 + 2.0*np.exp(-(3.0*t/T)**2)),
                "S3":(lambda t, T: (-2.0 * t**3/T**3) + 2.5),
                "S4":(lambda t, T: 1.5 * np.sqrt(2.5 - (2.0*np.log(t)/np.log(T)))) 
            },
    "TCSSA5":
            {
                "S1":(lambda t, T: np.sqrt(-2.05 * t / T + 2.55)),
                "S2":(lambda t, T: 1.95 - 2.0*t**(1.0/3)/T**(1.0/3)),
                "S3":(lambda t, T: -2.05 * t / T + 2.55),
                "S4":(lambda t, T: (-2.0 * t**3/T**3) + 2.5)  
            }
}

if __name__ == '__main__':
    T = 200
    fig = plt.figure()
    l = 1
    for st, c, name in zip(UPDATE_STRATEGIES, ['r', 'g', 'b', 'y', 'c'], ["M1", "M2", "M3", "M4", "M5"]):
        for sb in sorted(UPDATE_STRATEGIES[st]):
            f = UPDATE_STRATEGIES[st][sb]
            s = [f(t, T) for t in range(T)]
            ax = fig.add_subplot(len(UPDATE_STRATEGIES.keys()), 4, l)
            l = l+1
            ax.plot(s, c, label=name+"-"+sb)
            ax.legend()
        T = 200
    fig = plt.figure()
    l = 1
    for st, c, name in zip(["TCSSA1", "TCSSA4", "TCSSA5"], ['r', 'g', 'b'], ["M1", "M2", "M3"]):
        for sb in sorted(UPDATE_STRATEGIES[st]):
            f = UPDATE_STRATEGIES[st][sb]
            s = [f(t, T) for t in range(T)]
            ax = fig.add_subplot(3, 4, l)
            l = l+1
            ax.plot(s, c, label=name+"-"+sb)
            ax.legend()
    plt.show()
