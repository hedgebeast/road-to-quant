import numpy as np
import matplotlib.pylab as plt


"""
KEY: 
    W_{t+\Delta t} - W_t ~ N(0, \Delta t) ~ \sqrt{\Delta t} N(0, 1)
  
"""

def simulate_Brownian_Motion(paths, steps, T):
    deltaT = T/steps
    t = np.linspace(0, T, steps+1)
    
#     X = np.c_[np.zeros((paths, 1)),
#               np.random.randn(paths, steps)]

    X = np.hstack([np.zeros((paths, 1)),
          np.random.randn(paths, steps)])
        
    return t, np.cumsum(np.sqrt(deltaT) * X, axis=1)

# plt.figure()
# t, x = simulate_Brownian_Motion(5000, 100, 10.0)
# plt.plot(t, x.T)
# plt.show()



def simulate_Random_Walk(n, bernoulli = [-1, 1]):
    
    # Define parameters for the walk
    dims = 1
    step_n = n
    step_set = bernoulli
    origin = np.zeros((1,dims))

    step_shape = (step_n,dims)
    # the sample assumes a uniform distribution over all entries in a below
    steps = np.random.choice(a=step_set, size=step_shape) 
    path = np.concatenate([origin, steps]).cumsum()
    start = path[:1]
    stop = path[-1:]
    
    # Plot the path
    # fig = plt.figure(figsize=(8,4),dpi=200)
    # ax = fig.add_subplot(111)
    # ax.scatter(np.arange(step_n+1), path, c='blue',alpha=0.25,s=0.05)
    # ax.plot(path,c='blue',alpha=0.5,lw=0.5,ls='-')
    # ax.plot(0, start, c='green', marker='+')
    # ax.plot(step_n, stop, c='red', marker='+')
    # plt.title('1D Random Walk')
    # plt.tight_layout(pad=0)
    # plt.show()
    