import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


# define lorenz system
def lorenz_system(state, t):
    x, y, z = state
    a = 10
    b = 35
    c = 8.0 / 3.0
    return [a * (y - x), x * (b - z) - y, x * y - c * z]


# simulation parameter
T_prepare = 1000
T_train = 2000
delta_T = T_prepare + T_train
T_pred = 1250
dt = 0.01
N = 500
input_dimension = 3
sigma = 1
alpha_values = 0.53
beta = 0.477
pc = 0.95
lamda = 1e-7


class ReservoirComputing:
    def __init__(
        self,
        num_neuron=N,
        sigma=sigma,
        alpha=alpha_values,
        beta=beta,
        dt=dt,
        Din=input_dimension,
    ):
        self.num_neuron = num_neuron
        self.sigma = sigma
        self.beta = beta
        self.Din = Din
        self.dt = dt
        self.alpha = alpha
        self.Win = np.random.uniform(-sigma, sigma, (num_neuron, Din))
        self.matrix_A = np.random.rand(num_neuron, num_neuron)
        self.matrix_B = np.random.rand(num_neuron, 1)
        self.matrix_H = np.random.rand(num_neuron, Din)
        self.matrix_w = np.random.uniform(-sigma, sigma, (num_neuron, 1))

        self.state = np.zeros(self.num_neuron)

    def theta_process(self, input, timesteps):
        N = self.num_neuron
        alpha = self.alpha
        self.theta = np.random.uniform(0, 1, N)
        self.theta_list = []

        for t in range(timesteps):
            delta_theta = self.theta[:, np.newaxis] - self.theta
            sum_theta = -np.sum(self.matrix_A * np.sin(delta_theta), axis=1)

            sum_U = np.dot(self.matrix_H, input[t, :])

            theta_dot = (
                (1 - alpha) * self.matrix_w.flatten()
                + (alpha / N) * sum_theta
                + self.beta * np.tanh(self.matrix_B.flatten() + sum_U)
            )

            self.theta += theta_dot * self.dt

            # self.theta = np.unwrap(self.theta)
            self.theta = np.mod(self.theta, 2 * np.pi)

            self.theta_list.append(np.copy(self.theta))

        self.theta_list = np.array(self.theta_list)
        return self.theta_list

    def train(self, inputs, state_history):
        self.model = Ridge(alpha=self.beta, fit_intercept=False)
        self.model.fit(state_history[:], inputs[:])

        training_loss = mean_squared_error(
            inputs[:], self.model.predict(state_history)[:]
        )
        print(f"Training Loss (MSE): {training_loss}")

        return self.model.predict(state_history)


initial_state = np.array([1.0, 1.0, 1.0])
time = np.arange(0, T_train + T_pred) * dt
time_train, time_test = time[:T_train], time[T_train:]
data = odeint(lorenz_system, initial_state, time)
data_train, data_test = (
    data[T_prepare : T_prepare + T_train],
    data[T_prepare + T_train :],
)


RC = ReservoirComputing()
reservoir_states = RC.theta_process(data, delta_T)

print(reservoir_states.shape)
training_states = RC.train(data_train, reservoir_states[T_prepare:])

# #predict
# prediction, state = RC.prediction(data_test.shape[0])
#
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(time_train, data_train[:, 0], label="Train Data")
# plt.plot(time_train, reservoir_states[T_prepare : T_prepare + T_train, 0], label="Reservoir")
plt.plot(time_train, training_states[:, 0], label="Test")
# plt.plot(time_test, state[:, 0], label="state")
# plt.plot(time_test, data_test[:, 0], label="True")
plt.ylabel("x")
#
# plt.subplot(3,1,2)
# plt.plot(time_train, data_train[:, 1], label="Train Data")
# plt.plot(time_train, reservoir_states[:, 1], label="Reservoir")
# plt.plot(time_test, prediction[:, 1], label="Test")
# plt.plot(time_test, data_test[:, 1], label="True")
# plt.ylabel('y')
#
# plt.subplot(3,1,3)
# plt.plot(time_train, data_train[:, 2], label="Train Data")
# plt.plot(time_train, reservoir_states[:, 2], label="Reservoir")
# plt.plot(time_test, prediction[:, 2], label="Test")
# plt.plot(time_test, data_test[:, 2], label="True")
# plt.ylabel('z')
# plt.xlabel('t')
#
# plt.legend()
plt.show()
