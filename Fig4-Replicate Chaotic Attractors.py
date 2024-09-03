import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Solution to KS system
# KSequ.m - solution of Kuramoto-Sivashinsky equation
#
# u_t = -u*u_x - u_xx - u_xxxx, periodic boundary conditions on [0,32*pi]
# computation is based on v = fft(u), so linear term is diagonal
#
# Using this program:
# u is the initial condition
# h is the time step
# N is the number of points calculated along x
# a is the max value in the initial condition
# b is the min value in the initial condition
# x is used when using a periodic boundary condition, to set up in terms of
#   pi
#
# Initial condition and grid setup
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


class ReservoirComputing:
    def __init__(self, num_neuron=9000, spectral_radius=0.4, sigma=0.5, sparsity=0.98, beta=0.0001):
        self.num_neuron = num_neuron
        self.spectral_radius = spectral_radius
        self.sigma = sigma
        self.beta = beta
        self.Win = np.random.uniform(-sigma, sigma, (num_neuron, 1024))
        self.recurrent_matrix = np.random.rand(num_neuron, num_neuron) - 0.5
        self.recurrent_matrix[np.random.rand(*self.recurrent_matrix.shape) < sparsity] = 0
        rho_A = max(abs(np.linalg.eigvals(self.recurrent_matrix)))
        self.recurrent_matrix *= spectral_radius / rho_A

        self.state = np.zeros(self.num_neuron)

    def train(self, inputs):
        steps = inputs.shape[0]
        state_history = np.zeros((steps, self.num_neuron))
        for t in range(steps):
            self.state = np.tanh(np.dot(self.recurrent_matrix, self.state) + np.dot(self.Win, inputs[t]))
            state_history[t, :] = self.state

        self.model = Ridge(alpha=self.beta, fit_intercept=False)
        self.model.fit(state_history[:-1], inputs[1:])

        training_loss = mean_squared_error(inputs[1:], self.model.predict(state_history)[1:])
        print(f"Training Loss (MSE): {training_loss}")

        return self.model.predict(state_history)

    def prediction(self, steps):
        prediction = np.zeros((steps, 1024))
        state=np.zeros((steps, 9000))

        for t in range(steps):
            self.state = np.tanh(np.dot(self.recurrent_matrix, self.state) + np.dot(self.Win, self.model.predict(self.state[None,:])[0]))
            prediction[t, :] = self.model.predict(self.state[None,:])[0]
            state[t,:] = self.state

        return prediction, state

data_train, data_test = uu[:1000], uu[1000:]
time_train, time_test = tt[:1000], tt[1000:]
RC = ReservoirComputing()
reservoir_states = RC.train(data_train)
prediction, state = RC.prediction(data_test.shape[0])

# plot
# 拼接 reservoir_states 和 prediction 在时间轴上的数据
combined_reservoir_states = np.hstack((reservoir_states.transpose(), prediction.transpose()))
difference = uu.transpose() - combined_reservoir_states
separator_position = reservoir_states.shape[0] * h  # reservoir_states 在时间轴的长度
fig, ax = plt.subplots(3, 1, figsize=(12, 8))

im1 = ax[0].imshow(uu.transpose(), aspect='auto', origin='lower', cmap='jet', vmin=-2, vmax=2, extent=[0, tt[-1], 0, 60]) # the second parameter will be x-axis of the figure
ax[0].set_title("KS System")
fig.colorbar(im1, ax=ax[0])

im2 = ax[1].imshow(combined_reservoir_states, aspect='auto', origin='lower', cmap='jet', vmin=-2, vmax=2, extent=[0, tt[-1], 0, 60])
ax[1].set_title("Reservoir States and Prediction Combined")
fig.colorbar(im2, ax=ax[1])

im3 = ax[2].imshow(difference, aspect='auto', origin='lower', cmap='jet', vmin=-2, vmax=2, extent=[0, tt[-1], 0, 60])
ax[2].set_title("Difference")
fig.colorbar(im3, ax=ax[2])

ax[1].axvline(x=separator_position, color='red', linestyle='--', linewidth=1)
ax[2].axvline(x=separator_position, color='red', linestyle='--', linewidth=1)
plt.tight_layout()  # 自动调整子图布局
plt.show()
