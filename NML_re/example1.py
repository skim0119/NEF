import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import random as sparse_random
from scipy.sparse.linalg import eigs
from scipy.linalg import lstsq
from scipy.signal import firwin, lfilter
from sklearn.linear_model import Ridge
from numpy.linalg import norm


from ReservoirTanhB import ReservoirTanhB
from decomp_poly4_ns import decomp_poly4_ns
from Thomas import Thomas

np.random.seed(0)

#Reservoir parameters
dt = 0.001
gam = 100
num_neuron = 1000
sparsity = 0.05
recurrent_matrix = np.random.rand(num_neuron, num_neuron) - 0.5
recurrent_matrix[np.random.rand(*recurrent_matrix.shape) > sparsity] = 0
"""
A1 = recurrent_matrix * 0.01
B1 = np.random.rand(num_neuron, 3) - 0.5
rs1 = np.random.rand(num_neuron, 1) - 0.5
"""
A1 = np.loadtxt('originaldata/A1.txt',delimiter=',')
B1 = np.loadtxt('originaldata/B1.txt',delimiter=',')
rs1 = np.loadtxt('originaldata/rs1.txt',delimiter=',')
rs1 = rs1[:, np.newaxis]
xs1 = np.array([0, 0, 0]).reshape(3, 1)

R1 = ReservoirTanhB(A1,B1,rs1,xs1,dt,gam)  # RNN class
R1.r = rs1
d1 = R1.d

_, C1, C1a, *_ = decomp_poly4_ns(A1, B1, rs1, A1 @ rs1 + B1 @ xs1 + d1, gam, 3) # Polynomial expansion

RsNPL1 = np.hstack([C1, C1a.reshape(num_neuron, C1a.shape[1] * 3)])
# initialize OsNPL1 and rotational matrix
Rz = np.array([[0, -1, 0],
               [1,  0, 0],
               [0,  0, 1]])

Rz = np.array([[0, -1, 0],
               [1,  0, 0],
               [0,  0, 1]])
OsNPL1 = np.zeros((3, C1.shape[1] * 4))
OsNPL1[:, 1:4] = Rz

# residual solution
W1, _, _, _ = lstsq(RsNPL1.T, OsNPL1.T, lapack_driver='gelsy')
W1 = W1.T

# calculate residual
residual = norm(W1 @ RsNPL1 - OsNPL1, 1)
print("Compiler residual:", residual)

# EXAMPLE: Decompile, program, and compile rotation for continuous-time
# Generate input and drive RNN
n = 100000
T = Thomas(np.array([0, 0, 1]), dt)  # Thomas attractor object
X1 = T.propagate(n)
Ro1 = R1.train(X1)  # get the reservoir states in each time steps
X1 = X1[:, int(0.2*n):, 0]
Ro1 = Ro1[:, int(0.2*n):]
WR1 = W1 @ Ro1
XRot = Rz @ X1 # desired output

# calculate rotation error
error = norm(XRot - WR1)/norm(XRot - X1)
print("Reletive error:", error)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(X1[0, :], X1[1, :], X1[2, :], lw=0.5, color = "blue", label = 'Input data')
ax.plot(WR1[0, :], WR1[1, :], WR1[2, :], lw=0.5, color = "red", label = 'Output data')
ax.plot(XRot[0, :], XRot[1, :], XRot[2, :], lw=0.5, color = "yellow", label = 'Expected output data')

ax.set_title("Thomas Attractor")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.legend()
plt.show()




