import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.sparse import random as sparse_random
from scipy.sparse.linalg import eigs
from scipy.linalg import lstsq
from scipy.signal import firwin, lfilter
from sklearn.linear_model import Ridge
from numpy.linalg import norm

from ReservoirTanhB import ReservoirTanhB
from decomp_poly_ns_discrete import decomp_poly_ns_discrete
from Thomas import Thomas


np.random.seed(0)

# Reservoir parameters
dt = 0.001
gam = 100
sparsity = 0.05
Num_neuron = 1000

# A2 = np.random.rand(Num_neuron, Num_neuron) - 0.5
# A2[np.random.rand(Num_neuron, Num_neuron) > sparsity] = 0
# B2 = np.random.rand(Num_neuron, 1) - 0.5
# rs2 = np.random.rand(Num_neuron, 1) - 0.5


A2 = np.loadtxt("originaldata/A2.txt", delimiter=",")
B2 = np.loadtxt("originaldata/B2.txt", delimiter=",")
B2 = B2.reshape(Num_neuron, 1)
rs2 = np.loadtxt("originaldata/rs2.txt", delimiter=",")
rs2 = rs2.reshape(Num_neuron, 1)
xs2 = np.array([0]).reshape(1, 1)
d2 = np.arctanh(rs2) - A2 @ rs2

# decompile into SNPL
[Pd2, C2] = decomp_poly_ns_discrete(A2, B2, rs2, A2 @ rs2 + B2 @ xs2 + d2, 4, 3)

RsNPL2 = np.hstack(
    [C2[:, 0, 0].reshape(Num_neuron, 1), C2[:, 1:, 0], C2[:, 1:, 1], C2[:, 1:, 2]]
)  ##### Different from matlab

# Initialize variables
OsNPL2 = np.zeros((3, RsNPL2.shape[1]))

# Create 3 high-pass filters with different cutoff frequencies
# of = np.array([signal.firwin(3, 0.1, pass_zero=False),  # 高通滤波器
#                signal.firwin(3, 0.2, pass_zero=False),
#                signal.firwin(3, 0.3, pass_zero=False)])       # the filter in python is not working

of = np.array(
    (
        [-0.0896879756980144, 0.820624048603971, -0.0896879756980144],
        [-0.159341287490797, 0.681317425018406, -0.159341287490797],
        [-0.211942742334250, 0.576114515331499, -0.211942742334250],
    )
)


# Assign the filter coefficients to specific columns of OsNPL2
OsNPL2[0:3, 1::3] = of

# Least squares solution (similar to lsqminnorm)
W2, _, _, _ = lstsq(RsNPL2.T, OsNPL2.T)
W2 = W2.T


# Generate input and drive RNN
t = np.linspace(0, 3, 91)
X2 = np.sin(t * 2 * np.pi) + 0.2 * np.sin(t * 8 * np.pi)

# Assuming A2, B2, d2, and rs2 are defined
Ro2 = np.zeros((Num_neuron, len(t)))
rs2 = np.squeeze(rs2)
Ro2[:, 0] = rs2

for i in range(1, len(t)):
    Ro2[:, i] = np.squeeze(
        np.tanh(
            A2 @ Ro2[:, i - 1].reshape(Num_neuron, 1)
            + B2 @ X2[i - 1].reshape(1, 1)
            + d2
        )
    )

# Compute WR2
WR2 = W2 @ Ro2
# Filtering the input signal X2 using the designed high-pass filters
XFilt = np.zeros((3, Ro2.shape[1] - 2))
for i in range(XFilt.shape[1]):
    XFilt[:, i] = np.dot(of, X2[i : i + 3])

print(W2.shape)
# Calculate relative error
errv = np.linalg.norm(XFilt[:, :-1] - WR2[:, 3:]) / np.linalg.norm(XFilt[:, :-1])
print(f"Relative error: {errv}")

fig = plt.figure()
plt.plot(t, WR2[0, :], label="Neuron output 1")
plt.plot(t, WR2[1, :], label="Neuron output 2")
plt.plot(t, WR2[2, :], label="Neuron output 3")
plt.plot(t, X2, label="Input")
plt.xlabel("t")


# ax[2].plot(t, WR2-XFilt, label = "Neuron output")

plt.legend()
plt.show()
