import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.linear_model import Ridge
from scipy.signal import argrelextrema

#define lorenz system
def lorenz_system(state, t):
    x, y, z = state
    a = 10
    b = 28
    c = 8.0/3.0
    return [a*(y-x), x*(b-z)-y, x*y-c*z]

#define RC
class ReservoirComputing:
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

    def train(self, inputs):
        steps = inputs.shape[0]
        state_history = np.zeros((steps, self.num_neuron))
        for t in range(steps):
            self.state = np.tanh(np.dot(self.recurrent_matrix, self.state) + np.dot(self.Win, inputs[t]))
            state_history[t, :] = self.state

        self.model = Ridge(alpha=self.beta, fit_intercept=False)
        self.model.fit(state_history[:-1], inputs[1:])

        return self.model.predict(state_history)

    def prediction(self, steps):
        prediction = np.zeros((steps, 3))

        for t in range(steps):
            self.state = np.tanh(np.dot(self.recurrent_matrix, self.state) + np.dot(self.Win, self.model.predict(self.state[None,:])[0]))

            prediction[t, :] = self.model.predict(self.state[None,:])[0]

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

#train
reservoir_states = RC.train(data_train)

#predict
prediction = RC.prediction(data_test.shape[0])

test_maxima_indices = argrelextrema(data_test[:, 2], np.greater)[0]
predicted_maxima_indices = argrelextrema(prediction[:, 2], np.greater)[0]

test_maxima = data_test[test_maxima_indices, 2]
predicted_maxima = prediction[predicted_maxima_indices, 2]

#plot
fig, ax = plt.subplots(2, 1, figsize = (10, 6))
ax[0].scatter(test_maxima[:-1], test_maxima[1:], c = 'blue', label="Actual Maxima")
ax[0].scatter(predicted_maxima[:-1], predicted_maxima[1:], c = 'red', label="Predicted Maxima")
ax[0].set_xlim(25,50)
ax[0].set_ylim(25,50)
ax[0].set_xlabel(r'$z_i$')
ax[0].set_ylabel(r'$z_{z+1}$')


plt.show()