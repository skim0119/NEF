import numpy as np
import matplotlib.pyplot as plt

N = 500
input_dimension = 1
sigma = 1
alpha_values = 0.53
beta = 0
dt = 0.01
delta_T = 260
pc = 0.95
class ReservoirComputing:
    def __init__(self, num_neuron=N, sigma=sigma, alpha=alpha_values, delta_T=delta_T, beta=beta, dt=dt, Din=input_dimension):
        self.num_neuron = num_neuron
        self.sigma = sigma
        self.beta = beta
        self.Din = Din
        self.dt = dt
        self.alpha = alpha
        self.delta_T = delta_T
        self.Win = np.random.uniform(-sigma, sigma, (num_neuron, Din))
        self.matrix_A = np.random.rand(num_neuron, num_neuron)
        self.matrix_B = np.random.rand(num_neuron, 1)
        self.matrix_H = np.random.rand(num_neuron, Din)
        self.matrix_w = np.random.uniform(-sigma, sigma, (num_neuron, 1))

        self.state = np.zeros(self.num_neuron)

    def theta_process(self, input):
        N = self.num_neuron
        timesteps = self.delta_T
        alpha = self.alpha
        self.theta = np.random.uniform(-1, 1, N)
        self.theta_list = []

        for t in range(timesteps):
            delta_theta = self.theta[:, np.newaxis] - self.theta
            sum_theta = -np.sum(self.matrix_A * np.sin(delta_theta), axis=1)

            sum_U = 0
            # sum_U = np.dot(self.matrix_H, input)

            theta_dot = (1 - alpha) * self.matrix_w.flatten() + (alpha / N) * sum_theta + self.beta * np.tanh(
                self.matrix_B.flatten() + sum_U)

            self.theta += theta_dot * self.dt

            #self.theta = np.unwrap(self.theta)
            self.theta = np.mod(self.theta, 2 * np.pi)

            self.theta_list.append(np.copy(self.theta))

        theta = self.theta_list

        return self.theta

    def correlation_process(self):
        theta_array = np.array(self.theta_list)
        correlation_matrix = np.corrcoef(theta_array.T)

        return correlation_matrix

    def binary_process(self, P, pc):
        N = self.num_neuron
        for i in range(N):
            for j in range(N):
                if P[i][j] < pc:
                    P[i][j] = 0
        return P


RC = ReservoirComputing()
theta = RC.theta_process(np.zeros(1))
p = RC.correlation_process()

pcopy = np.copy(p)
B = RC.binary_process(pcopy,pc)

plt.subplot(1, 2, 1)
plt.imshow(p, cmap='jet', interpolation='nearest', vmin=-1, vmax=1)
plt.colorbar(label='Correlation Coefficient', shrink = 0.5)
plt.title('Pairwise Correlation Matrix')
plt.xlabel(r'Neuron $i$')
plt.ylabel(r'Neuron $j$')
plt.gca().invert_yaxis()
plt.grid(False)
plt.subplot(1, 2, 2)
plt.imshow(B, cmap='jet', interpolation='nearest', vmin=-1, vmax=1)
plt.colorbar(label='Correlation Coefficient', shrink = 0.5)
plt.title('Pairwise Correlation Matrix')
plt.xlabel(r'Neuron $i$')
plt.ylabel(r'Neuron $j$')
plt.gca().invert_yaxis()
plt.grid(False)

plt.show()

