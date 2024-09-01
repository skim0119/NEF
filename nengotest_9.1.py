import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import nengo
from nengo.solvers import LstsqL2
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

#define lorenz system
def lorenz_system(state, t):
    x, y, z = state
    a = 10
    b = 28
    c = 8.0/3.0
    return [a*(y-x), x*(b-z)-y, x*y-c*z]

#define RC using Nengo
class NengoReservoirComputing:
    def __init__(self, num_neuron=300, spectral_radius=1.2, sigma=0.1, sparsity=0.98, beta=0.0):
        self.num_neuron = num_neuron
        self.spectral_radius = spectral_radius
        self.sigma = sigma
        self.beta = beta
        self.Win = np.random.uniform(-sigma, sigma, (num_neuron, 3))
        self.recurrent_matrix = np.random.rand(num_neuron, num_neuron) - 0.5
        self.recurrent_matrix[np.random.rand(*self.recurrent_matrix.shape) < sparsity] = 0
        rho_A = max(abs(np.linalg.eigvals(self.recurrent_matrix)))
        self.recurrent_matrix *= spectral_radius / rho_A

        self.state = np.zeros(self.num_neuron)

        # Nengo model setup
        self.model = nengo.Network()
        with self.model:
            # Input node
            self.input = nengo.Node(size_in=3)

            # Reservoir neurons
            self.reservoir = nengo.Ensemble(n_neurons=num_neuron, dimensions=num_neuron, neuron_type=nengo.Direct())

            # Connect input to reservoir
            nengo.Connection(self.input, self.reservoir,
                             transform=self.Win, synapse=None)

            # Recurrent connections in the reservoir
            nengo.Connection(self.reservoir, self.reservoir,
                             transform=self.recurrent_matrix, synapse=0,
                             function=np.tanh)

            # Probes to collect reservoir states
            self.reservoir_probe = nengo.Probe(self.reservoir, synapse=0)

    def train(self, inputs, dt=0.02):
        steps = inputs.shape[0]
        is_training = [True]

        def input_func(t):
            if is_training[0] and int(t / dt) < steps:
                return inputs[int(round(t / dt))]
            else:
                return [0, 0, 0]

        with self.model:
            self.input_data = nengo.Node(output=input_func)
            nengo.Connection(self.input_data, self.input)

        # Run the model
        with nengo.Simulator(self.model, dt=dt) as sim:
            sim.run(steps * dt)

        reservoir_states = sim.data[self.reservoir_probe]
        self.Lmodel = Ridge(alpha=self.beta, fit_intercept=False)
        self.Lmodel.fit(reservoir_states[:-1], inputs[1:])
        self.last_state = reservoir_states[-1]
        is_training[0] = False

        training_loss = mean_squared_error(inputs[1:], self.Lmodel.predict(reservoir_states)[1:])
        print(f"Training Loss (MSE): {training_loss}")

        return self.Lmodel.predict(reservoir_states)


    def prediction(self, steps, dt=0.02):
        current_state = self.last_state

        with self.model:
            def input_func(t):
                if t == dt:
                    # 在 t=0 时使用初始状态进行预测
                    return self.Lmodel.predict(current_state[None, :])[0]
                else:
                    # 其余时间返回零输入
                    return [0, 0, 0]

            self.input_data = nengo.Node(output=input_func)

            nengo.Connection(self.input_data, self.input, synapse=dt)
            nengo.Connection(self.reservoir, self.input, transform=self.Lmodel.coef_, synapse=0)
            self.prediction_probe = nengo.Probe(self.input, synapse=None)
            self.input_probe = nengo.Probe(self.reservoir, synapse=None)

        with nengo.Simulator(self.model, dt=dt) as sim:
            sim.run(steps * dt)

        predictions = sim.data[self.prediction_probe]
        input = sim.data[self.input_probe]
        return predictions, input


# Simulation parameters
T_train = 5000
T_pred = 10
dt = 0.02
initial_state = np.array([1.0, 1.0, 1.0])
time = np.arange(0, T_train + T_pred) * dt
time_train, time_test = time[:T_train], time[T_train:]

data = odeint(lorenz_system, initial_state, time)
data_train, data_test = data[:T_train], data[T_train:]

RC = NengoReservoirComputing()

# Train
reservoir_states = RC.train(data_train, dt=dt)

# Predict
prediction, input = RC.prediction(data_test.shape[0], dt=dt)

# Plot
fig, ax = plt.subplots(3, 1, figsize = (10, 6))
ax[0].plot(time_train, data_train[:, 0], label="Train Data")
ax[0].plot(time_train, reservoir_states[:, 0], label="Reservoir")
ax[0].plot(time_test, prediction[:, 0], label="Test")
ax[0].plot(time_test, input[:, 0], label="input")
ax[0].plot(time_test, data_test[:, 0], label="True")
ax[0].set_ylabel('x')

ax[1].plot(time_train, data_train[:, 1], label="Train Data")
ax[1].plot(time_train, reservoir_states[:, 1], label="Reservoir")
ax[1].plot(time_test, prediction[:, 1], label="Test")
ax[1].plot(time_test, data_test[:, 1], label="True")
ax[1].set_ylabel('y')

ax[2].plot(time_train, data_train[:, 2], label="Train Data")
ax[2].plot(time_train, reservoir_states[:, 2], label="Reservoir")
ax[2].plot(time_test, prediction[:, 2], label="Test")
ax[2].plot(time_test, data_test[:, 2], label="True")
ax[2].set_ylabel('z')
ax[2].set_xlabel('t')

plt.legend()
plt.show()
