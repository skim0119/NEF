import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.linear_model import Ridge


# define lorenz system
def lorenz_system(state, t):
    x, y, z = state
    a = 10
    b = 28
    c = 8.0 / 3.0
    return [a * (y - x), x * (b - z) - y, x * y - c * z]


# define RC
class ReservoirComputing:
    def __init__(
        self, num_neuron=300, spectral_radius=1.2, sigma=0.1, sparsity=0.98, beta=0.0
    ):
        self.num_neuron = num_neuron
        self.spectral_radius = spectral_radius
        self.sigma = sigma
        self.beta = beta
        self.Win = np.random.uniform(-sigma, sigma, (num_neuron, 3))
        self.recurrent_matrix = np.random.rand(num_neuron, num_neuron) - 0.5
        self.recurrent_matrix[
            np.random.rand(*self.recurrent_matrix.shape) < sparsity
        ] = 0
        rho_A = max(abs(np.linalg.eigvals(self.recurrent_matrix)))
        self.recurrent_matrix *= spectral_radius / rho_A

        self.state = np.zeros(self.num_neuron)

    def train(self, inputs):
        steps = inputs.shape[0]
        state_history = np.zeros((steps, self.num_neuron))
        for t in range(steps):
            self.state = np.tanh(
                np.dot(self.recurrent_matrix, self.state) + np.dot(self.Win, inputs[t])
            )
            state_history[t, :] = self.state

        self.model = Ridge(alpha=self.beta, fit_intercept=False)
        self.model.fit(state_history[:-1], inputs[1:])

        return self.model.predict(state_history)

    def prediction(self, steps):
        prediction = np.zeros((steps, 3))

        for t in range(steps):
            self.state = np.tanh(
                np.dot(self.recurrent_matrix, self.state)
                + np.dot(self.Win, self.model.predict(self.state[None, :])[0])
            )

            prediction[t, :] = self.model.predict(self.state[None, :])[0]

        return prediction


# simulation parameter
T_train = 5000
T_pred = 1250
dt = 0.02
initial_state = np.array([1.0, 1.0, 1.0])
time = np.arange(0, T_train + T_pred) * dt
time_train, time_test = time[:T_train], time[T_train:]

data = odeint(lorenz_system, initial_state, time)
data_train, data_test = data[:T_train], data[T_train:]

print(data_train.shape)
print(data_test.shape)

RC = ReservoirComputing()

# train
reservoir_states = RC.train(data_train)

# predict
prediction = RC.prediction(data_test.shape[0])

# plot
fig, ax = plt.subplots(3, 1, figsize=(10, 6))
ax[0].plot(time_train, data_train[:, 0], label="Train Data")
ax[0].plot(time_train, reservoir_states[:, 0], label="Reservoir")
ax[0].plot(time_test, prediction[:, 0], label="Test")
ax[0].plot(time_test, data_test[:, 0], label="True")
ax[0].set_ylabel("x")

ax[1].plot(time_train, data_train[:, 1], label="Train Data")
ax[1].plot(time_train, reservoir_states[:, 1], label="Reservoir")
ax[1].plot(time_test, prediction[:, 1], label="Test")
ax[1].plot(time_test, data_test[:, 1], label="True")
ax[1].set_ylabel("y")

ax[2].plot(time_train, data_train[:, 2], label="Train Data")
ax[2].plot(time_train, reservoir_states[:, 2], label="Reservoir")
ax[2].plot(time_test, prediction[:, 2], label="Test")
ax[2].plot(time_test, data_test[:, 2], label="True")
ax[2].set_ylabel("z")
ax[2].set_xlabel("t")

plt.legend()
plt.show()
