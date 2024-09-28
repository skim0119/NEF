import numpy as np
import matplotlib.pyplot as plt
from lyapynov import ContinuousDS, DiscreteDS
from lyapynov import mLCE, LCE

# Definition of a continuous dynamical system, here Lorenz63.
sigma = 10.
rho = 28.
beta = 8./3.
x0 = np.array([1.5, -1.5, 20.])
t0 = 0.
dt = 1e-2

def f(x,t):
    res = np.zeros_like(x)
    res[0] = sigma*(x[1] - x[0])
    res[1] = x[0]*(rho - x[2]) - x[1]
    res[2] = x[0]*x[1] - beta*x[2]
    return res

def jac(x,t):
    res = np.zeros((x.shape[0], x.shape[0]))
    res[0,0], res[0,1] = -sigma, sigma
    res[1,0], res[1,1], res[1,2] = rho - x[2], -1., -x[0]
    res[2,0], res[2,1], res[2,2] = x[1], x[0], -beta
    return res

Lorenz63 = ContinuousDS(x0, t0, f, jac, dt)
Lorenz63.forward(10**6, False)

# Computation of LCE
LCE, history = LCE(Lorenz63, 3, 0, 10**4, True)
# Print mLCE
print(LCE)

# Plot of mLCE evolution
plt.figure(figsize = (10,6))
plt.plot(history[:5000])
plt.xlabel("Number of time steps")
plt.ylabel("LCE")
plt.title("Evolution of the LCE for the first 5000 time steps")
plt.show()