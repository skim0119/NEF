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
dt = 0.25
gam = 300
num_neuron = 2000
sparsity = 0.05
recurrent_matrix = np.random.rand(num_neuron, num_neuron) - 0.5
recurrent_matrix[np.random.rand(*recurrent_matrix.shape) > sparsity] = 0
A1 = recurrent_matrix * 0.01
B1 = np.random.rand(num_neuron, 3) - 0.5
rs1 = np.random.rand(num_neuron, 1) - 0.5
rs1 = rs1
xs1 = np.array(np.zeros([1024, 1])).reshape(1024, 1)

R1 = ReservoirTanhB(A1,B1,rs1,xs1,dt,gam)  # RNN class
R1.r = rs1
d1 = R1.d

_, C1, C1a, *_ = decomp_poly4_ns(A1, B1, rs1, A1 @ rs1 + B1 @ xs1 + d1, gam, 3, 1024) # Polynomial expansion

RsNPL1 = np.hstack([C1, C1a.reshape(num_neuron, C1a.shape[1] * 3)])
print(RsNPL1.shape)
# initialize OsNPL1 and rotational matrix

N = 1024
x = np.linspace(0, 60, N)
a = -1
b = 1
u = 0.5*np.cos(x/16)*(1+np.sin(x/16))
v = np.fft.fft(u)
# scalars for ETDRK4
h = 0.25
k = np.transpose(np.conj(np.concatenate((np.arange(0, N/2), np.array([0]), np.arange(-N/2+1, 0))))) / 16
L = k**2 - k**4
E = np.exp(h*L)
E_2 = np.exp(h*L/2)
M = 16
r = np.exp(1j*np.pi*(np.arange(1, M+1)-0.5) / M)
LR = h*np.transpose(np.repeat([L], M, axis=0)) + np.repeat([r], N, axis=0)
Q = h*np.real(np.mean((np.exp(LR/2)-1)/LR, axis=1))
f1 = h*np.real(np.mean((-4-LR+np.exp(LR)*(4-3*LR+LR**2))/LR**3, axis=1))
f2 = h*np.real(np.mean((2+LR+np.exp(LR)*(-2+LR))/LR**3, axis=1))
f3 = h*np.real(np.mean((-4-3*LR-LR**2+np.exp(LR)*(4-LR))/LR**3, axis=1))
# main loop
uu = np.array([u])
tt = 0
tmax = 300
nmax = round(tmax/h)
nplt = 1
g = -0.5j*k
for n in range(1, nmax+1):
    t = n*h
    Nv = g*np.fft.fft(np.real(np.fft.ifft(v))**2)
    a = E_2*v + Q*Nv
    Na = g*np.fft.fft(np.real(np.fft.ifft(a))**2)
    b = E_2*v + Q*Na
    Nb = g*np.fft.fft(np.real(np.fft.ifft(b))**2)
    c = E_2*a + Q*(2*Nb-Nv)
    Nc = g*np.fft.fft(np.real(np.fft.ifft(c))**2)
    v = E*v + Nv*f1 + 2*(Na+Nb)*f2 + Nc*f3
    if n%nplt == 0:
        u = np.real(np.fft.ifft(v))
        uu = np.append(uu, np.array([u]), axis=0)
        tt = np.hstack((tt, t))

print(nmax)
print(uu.shape)
print(tt.shape)

OsNPL1 = np.zeros((N, uu.shape[0]-1))
OsNPL1 = uu

W1, _, _, _ = lstsq(RsNPL1.T, OsNPL1.T, lapack_driver='gelsy')
W1 = W1.T

"""
OsNPL1 = np.zeros((3, C1.shape[1] * 4))
OsNPL1[:, 1:4] = Rz

# residual solution


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

"""