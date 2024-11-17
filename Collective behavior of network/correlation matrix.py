import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm

# Simulation Parameters
N = 500
input_dimension = 1
sigma = 1
beta = 0
dt = 0.1
pc = 0.95


class ReservoirComputing:  # Computation setup
    def __init__(
        self, num_neuron=N, sigma=sigma, beta=beta, dt=dt, Din=input_dimension
    ):
        self.num_neuron = num_neuron
        self.sigma = sigma
        self.beta = beta
        self.Din = Din
        self.dt = dt
        self.Win = np.random.uniform(-sigma, sigma, (num_neuron, Din))
        self.matrix_A = np.random.rand(num_neuron, num_neuron)
        self.matrix_B = np.random.rand(num_neuron, 1)
        self.matrix_H = np.random.rand(num_neuron, Din)
        self.matrix_w = np.random.uniform(-sigma, sigma, (num_neuron, 1))

        self.state = np.zeros(self.num_neuron)

    def theta_process(self, input, alpha, timesteps):  # Compute theta
        N = self.num_neuron
        self.theta = np.random.uniform(-1, 1, N)
        self.theta_list = []
        r_values = []

        for t in range(timesteps):
            delta_theta = self.theta[:, np.newaxis] - self.theta
            sum_theta = -np.sum(
                self.matrix_A * np.sin(delta_theta), axis=1
            )  # calculate sum of a_ij * sin(theta_j(t) - theta_i(t)

            sum_U = 0
            # sum_U = np.dot(self.matrix_H, input)

            theta_dot = (
                (1 - alpha) * self.matrix_w.flatten()
                + (alpha / N) * sum_theta
                + self.beta * np.tanh(self.matrix_B.flatten() + sum_U)
            )  # calculate theta_dot

            self.theta += theta_dot * self.dt

            # self.theta = np.unwrap(self.theta)           # not sure if unwrap function can be used
            self.theta = np.mod(self.theta, 2 * np.pi)  # mod theta by 2*pi

            self.theta_list.append(np.copy(self.theta))

            r = np.abs(np.mean(np.exp(1j * self.theta)))
            r_values.append(r)

        r_time_avg = np.mean(r_values)  # calculate time average of r

        return self.theta, r_values, r_time_avg

    def correlation_process(self):  # correlation matrix calculation
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


# Simulation initiation
RC = ReservoirComputing()

# Calculate r versus a
# Initiate alpha and timesteps
alpha_values = np.linspace(0, 1.25, 100)
timesteps = 1000
r_values_alpha = []
r_time_avg_values = []
tind = 0

for alpha in alpha_values:
    if alpha > tind * np.max(alpha_values):
        print("=", end="", flush=True)  # Display progress
        tind += 0.01

    theta, r_values, r_time_avg = RC.theta_process(np.zeros(1), alpha, timesteps)
    r_values_alpha.append(r_values)
    r_time_avg_values.append(r_time_avg)


# Calculate correlation matrix
# Initiate alpha and timesteps
alpha_values_co = 0.53
timesteps = 260
theta = RC.theta_process(np.zeros(1), alpha_values_co, timesteps)
p = RC.correlation_process()


# Calculate clusters
pcopy = np.copy(p)
B = RC.binary_process(pcopy, pc)


# plot r versus a
plt.figure(1)
plt.plot(alpha_values, r_time_avg_values, "k-", marker="s", markersize=5)
plt.xlabel(r"$\alpha$", fontsize=14)
plt.ylabel(r"$r$", fontsize=14)
plt.title("Variation of Time-Averaged Order Parameter r with respect to Alpha")
plt.grid(True)


# plot correlation matrix
plt.figure(2)
plt.imshow(p, cmap="jet", interpolation="nearest", vmin=-1, vmax=1)
plt.colorbar(label="Correlation Coefficient", shrink=0.5)
plt.title("Pairwise Correlation Matrix")
plt.xlabel(r"Neuron $i$")
plt.ylabel(r"Neuron $j$")
plt.gca().invert_yaxis()
plt.grid(False)


# plot clusters
cmap = LinearSegmentedColormap.from_list("gray_red", ["lightgray", "red"])
plt.figure(3)
plt.imshow(B, cmap=cmap, interpolation="nearest", vmin=0, vmax=1)
plt.colorbar(label="Correlation Coefficient", shrink=0.5)
plt.title("Pairwise Correlation Matrix")
plt.xlabel(r"Neuron $i$")
plt.ylabel(r"Neuron $j$")
plt.gca().invert_yaxis()
plt.grid(False)
plt.show()
