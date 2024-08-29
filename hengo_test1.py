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

# RC
model = nengo.Network()
with model:
    # Input node
    stim = nengo.Node(lambda t: data[int(t / dt)] if t < T_train * dt else [0, 0, 0])

    # Reservoir
    reservoir = nengo.Ensemble(n_neurons=num_neuron, dimensions=3)

    # Manually set recurrent connection weights
    nengo.Connection(reservoir.neurons, reservoir.neurons, transform=recurrent_matrix, synapse=0.1)

    # Manually set input connection weights
    nengo.Connection(stim, reservoir.neurons, transform=Win, synapse=0.1)

    # Probes for data collection
    reservoir_probe = nengo.Probe(reservoir.neurons, synapse=0.1)

with nengo.Simulator(model, dt=0.02) as sim:
    sim.run(time_train[-1])

# Extract data recorded by the probe during training
reservoir_states = sim.data[reservoir_probe]
#print(prediction.shape)

# Train Ridge regression model using collected reservoir states
ridge_model = Ridge(alpha=beta, fit_intercept=False)
ridge_model.fit(reservoir_states[:], data_train[1:])
Wout = ridge_model.coef_

# Reservoir data
Reservoir_states = ridge_model.predict(reservoir_states)
reservoir_data = sim.data[reservoir_probe]

with model:
    # Input node
    reservoir = nengo.Ensemble(n_neurons=num_neuron, dimensions=3)
    nengo.Connection(reservoir.neurons, reservoir.neurons, transform=recurrent_matrix, synapse=0)

    # output node
    reservoir_output = nengo.Node(size_in=3)  # input signal dimensions: 3
    nengo.Connection(reservoir.neurons, reservoir_output, transform=Wout, synapse=0)
    nengo.Connection(reservoir_output, reservoir.neurons, transform=Win, synapse=0)

    # Probes for data collection
    reservoir_probe = nengo.Probe(reservoir.neurons, synapse=0.1)
    output_probe = nengo.Probe(reservoir_output, synapse=None)

with nengo.Simulator(model, dt=dt) as sim:
    sim.run(time_test[-1])

# Get prediction results
prediction = sim.data[output_probe]

# Plot results
fig, ax = plt.subplots(3, 1, figsize=(10, 6))
ax[0].plot(time_train, data_train[:, 0], label="Train Data")
ax[0].plot(time_test, data_test[:, 0], label="True")
ax[0].plot(time_train[1:], Reservoir_states[:T_train, 0], label="Reservoir States")
#ax[0].plot(time_test, prediction[-T_pred:, 0], label="Prediction")
ax[0].set_ylabel('x')

ax[1].plot(time_train, data_train[:, 1], label="Train Data")
ax[1].plot(time_test, data_test[:, 1], label="True")
ax[1].plot(time_train[1:], Reservoir_states[:T_train, 1], label="Reservoir States")
#ax[1].plot(time_test, prediction[-T_pred:, 1], label="Prediction")
ax[1].set_ylabel('y')

ax[2].plot(time_train, data_train[:, 2], label="Train Data")
ax[2].plot(time_test, data_test[:, 2], label="True")
ax[2].plot(time_train[1:], Reservoir_states[:T_train, 2], label="Reservoir States")
#ax[2].plot(time_test, prediction[-T_pred:, 2], label="Prediction")
ax[2].set_ylabel('z')
ax[2].set_xlabel('t')

plt.legend()
plt.show()


'''
fig, ax = plt.subplots(3, 1, figsize=(10, 6))
ax[0].plot(time[1:], stim_output[:, 0], label="Stim x")
ax[0].set_ylabel('x')
ax[0].legend()

ax[1].plot(time[1:], stim_output[:, 1], label="Stim y")
ax[1].set_ylabel('y')
ax[1].legend()

ax[2].plot(time[1:], stim_output[:, 2], label="Stim z")
ax[2].set_ylabel('z')
ax[2].legend()
ax[2].set_xlabel('t')

plt.show()
'''



