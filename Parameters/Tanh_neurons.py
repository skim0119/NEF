import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


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

        training_loss = mean_squared_error(
            inputs[1:], self.model.predict(state_history)[1:]
        )
        print(f"Training Loss (MSE): {training_loss}")

        return state_history, self.model.predict(state_history)

    def prediction(self, steps):
        prediction = np.zeros((steps, 3))
        test_state = np.zeros((steps, 300))

        for t in range(steps):
            self.state = np.tanh(
                np.dot(self.recurrent_matrix, self.state)
                + np.dot(self.Win, self.model.predict(self.state[None, :])[0])
            )
            prediction[t, :] = self.model.predict(self.state[None, :])[0]
            test_state[t, :] = self.state

        return test_state, prediction


# simulation parameter
T_train = 5000
T_pred = 1250
dt = 0.02
initial_state = np.array([1.0, 1.0, 1.0])
time = np.arange(0, T_train + T_pred) * dt
time_train, time_test = time[:T_train], time[T_train:]

data = odeint(lorenz_system, initial_state, time)
data_train, data_test = data[:T_train], data[T_train:]

RC = ReservoirComputing()
# train
training_states, reservoir_states = RC.train(data_train)
print(training_states.shape)

# predict
test_states, prediction = RC.prediction(data_test.shape[0])

# kernel size:
K = np.dot(training_states.T, training_states)  # K = (300, 300)
_, singular_values, _ = np.linalg.svd(K)  # SVD
threshold = 1e-5
kernel_rank = np.sum(singular_values > threshold)
print("Kernel Rank:", kernel_rank)

# Generalization size:
K = np.dot(test_states.T, test_states)  # K = (300, 300)
_, singular_values, _ = np.linalg.svd(K)  # SVD
threshold = 1e-5
Generalization_rank = np.sum(singular_values > threshold)
print("Generalization Rank:", Generalization_rank)

# Memory capacity
max_delay = 100
memory_capacity = 0
for k in range(1, max_delay + 1):
    X = training_states[:-k]  # reservoir states
    y = data_train[k:]  # input with k delay

    model = Ridge(alpha=0, fit_intercept=False)  # 使用 Ridge 回归避免过拟合
    model.fit(X, y)
    y_pred = model.predict(X)

    r2 = r2_score(y, y_pred)
    memory_capacity += r2
    print(f"Delay {k}: R^2 = {r2}")

print(f"Total Memory Capacity: {memory_capacity}")


plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(time_train, data_train[:, 0], label="Train Data")
plt.plot(time_train, reservoir_states[:, 0], label="Reservoir")
plt.plot(time_test, prediction[:, 0], label="Test")
plt.plot(time_test, data_test[:, 0], label="True")
plt.ylabel("x")

plt.subplot(3, 1, 2)
plt.plot(time_train, data_train[:, 1], label="Train Data")
plt.plot(time_train, reservoir_states[:, 1], label="Reservoir")
plt.plot(time_test, prediction[:, 1], label="Test")
plt.plot(time_test, data_test[:, 1], label="True")
plt.ylabel("y")

plt.subplot(3, 1, 3)
plt.plot(time_train, data_train[:, 2], label="Train Data")
plt.plot(time_train, reservoir_states[:, 2], label="Reservoir")
plt.plot(time_test, prediction[:, 2], label="Test")
plt.plot(time_test, data_test[:, 2], label="True")
plt.ylabel("z")
plt.xlabel("t")

plt.legend()
plt.show()
