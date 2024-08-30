import nengo
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.linear_model import Ridge


def lorenz_system(state, t):
    x, y, z = state
    a = 10
    b = 28
    c = 8.0/3.0
    return [a*(y-x), x*(b-z)-y, x*y-c*z]

# simulation parameter
T_train = 5000
T_pred = 1250
dt = 0.02
initial_state = np.array([1.0, 1.0, 1.0])
time = np.arange(0, T_train + T_pred) * dt
time_train, time_test = time[:T_train], time[T_train:]

data = odeint(lorenz_system, initial_state, time)
data_train, data_test = data[:T_train], data[T_train:]

# Reservoir Computing parameters
num_neuron = 300
spectral_radius = 1.2
sigma = 0.1
sparsity = 0.98
beta = 0.0  # Ridge regression parameter

# Create random recurrent weight matrix
recurrent_matrix = np.random.rand(num_neuron, num_neuron) - 0.5
recurrent_matrix[np.random.rand(*recurrent_matrix.shape) < sparsity] = 0  # Apply sparsity
rho_A = max(abs(np.linalg.eigvals(recurrent_matrix)))  # Compute spectral radius
recurrent_matrix *= spectral_radius / rho_A  # Adjust spectral radius

# Create input weight matrix
Win = np.random.uniform(-sigma, sigma, (num_neuron, 3))

class TikhonovRLS(nengo.learning_rules.RLS):
    def __init__(self, learning_rate, regularization, **kwargs):
        super(TikhonovRLS, self).__init__(learning_rate=learning_rate, **kwargs)
        self.regularization = regularization

    def update(self, w, pre, post, error, P):
        super(TikhonovRLS, self).update(w, pre, post, error, P)
        w -= self.learning_rate * self.regularization * w

# RC
model = nengo.Network()
with model:
    # Input node
    stim = nengo.Node(lambda t: data[int(round(t / dt))] if t < T_train * dt else [0, 0, 0])

    reservoir = nengo.Ensemble(n_neurons=num_neuron, dimensions=3)
    #I_R = nengo.Node(size_in = 300)
    R_O = nengo.Node(size_in = 3)
    eC = nengo.Node(size_in=3, output=lambda t, e: e if t < T_train else 0)
    data_test_nengo = nengo.Node(lambda t: data[int(t / dt)] if t >= T_train * dt else [0, 0, 0])

    nengo.Connection(stim, reservoir.neurons, synapse=None, transform=Win)
    nengo.Connection(reservoir.neurons, reservoir.neurons, synapse=0.1, transform=recurrent_matrix)
    conn = nengo.Connection(reservoir.neurons, R_O, synapse=None, transform=np.zeros((3, num_neuron)),
        learning_rule_type=TikhonovRLS(learning_rate=0.5, regularization=0.01))

    nengo.Connection(R_O, reservoir.neurons, synapse=0, transform=Win)
    nengo.Connection(R_O, eC, synapse=None)
    nengo.Connection(stim, eC, synapse=None, transform=-1)
    nengo.Connection(eC, conn.learning_rule)

    # Probes for data collection
    reservoir_probe = nengo.Probe(R_O, synapse=None)

with nengo.Simulator(model, dt=dt) as sim:
    sim.run(time_train[-1])

# Extract data recorded by the probe during training
reservoir_states = sim.data[reservoir_probe]


# Plot results
fig, ax = plt.subplots(3, 1, figsize=(10, 6))
ax[0].plot(time_train, data_train[:, 0], label="Train Data")
ax[0].plot(time_test, data_test[:, 0], label="True")
ax[0].plot(time_train[1:], reservoir_states[:T_train, 0], label="Reservoir States")
#ax[0].plot(time_test, prediction[-T_pred:, 0], label="Prediction")
ax[0].set_ylabel('x')

ax[1].plot(time_train, data_train[:, 1], label="Train Data")
ax[1].plot(time_test, data_test[:, 1], label="True")
ax[1].plot(time_train[1:], reservoir_states[:T_train, 1], label="Reservoir States")
#ax[1].plot(time_test, prediction[-T_pred:, 1], label="Prediction")
ax[1].set_ylabel('y')

ax[2].plot(time_train, data_train[:, 2], label="Train Data")
ax[2].plot(time_test, data_test[:, 2], label="True")
ax[2].plot(time_train[1:], reservoir_states[:T_train, 2], label="Reservoir States")
#ax[2].plot(time_test, prediction[-T_pred:, 2], label="Prediction")
ax[2].set_ylabel('z')
ax[2].set_xlabel('t')

plt.legend()
plt.show()





