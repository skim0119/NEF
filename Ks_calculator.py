import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

np.random.seed(0)


def ksintegrateNaive(u, Lx, dt, Nt, nplot):
    """
    Naive integration of the Kuramoto-Sivashinsky equation:
    u_t = -u*u_x - u_xx - u_xxxx, periodic boundary conditions
    """
    Nx = len(u)  # number of grid points
    kx = np.concatenate(
        [np.arange(0, Nx // 2), [0], np.arange(-Nx // 2 + 1, 0)]
    )  # integer wavenumbers
    alpha = 2 * np.pi * kx / Lx  # real wavenumbers: exp(i*alpha*x)
    D = 1j * alpha  # D = d/dx operator in Fourier space
    L = alpha**2 - alpha**4  # linear operator: -D^2 - D^4 in Fourier space
    G = -0.5 * D  # -1/2 D operator in Fourier space
    Nplot = Nt // nplot + 1  # total number of saved time steps

    x = np.linspace(0, Lx, Nx, endpoint=False)  # x values
    t = np.arange(Nplot) * dt * nplot  # time array
    U = np.zeros((Nplot, Nx))  # to store solution at saved time steps

    # some convenience variables
    dt2 = dt / 2
    dt32 = 3 * dt / 2
    A = 1 + dt2 * L
    B = 1 / (1 - dt2 * L)

    Nn = G * fft(u * u)  # -1/2 d/dx(u^2) = -u u_x (spectral calculation)
    Nn1 = Nn  # initialize N^{n-1}

    U[0, :] = u  # save initial condition to matrix U
    np_idx = 1  # counter for saved data

    u = fft(u)  # transform u to spectral

    # time-stepping loop
    for n in range(Nt):
        Nn1 = Nn  # shift N^{n-1} <- N^n
        u_real = np.real(ifft(u))  # convert u back to physical space
        Nn = G * fft(u_real**2)  # compute N^n = -u u_x

        # CNAB2 formula: B * (A * u + dt32 * Nn - dt2 * Nn1)
        u = B * (A * u + dt32 * Nn - dt2 * Nn1)

        if (n + 1) % nplot == 0:
            U[np_idx, :] = np.real(ifft(u))  # save solution at this time step
            np_idx += 1

    return U, x, t


# Parameters
Lx = 64
Nx = 1024
dt = 0.25
nplot = 1
Nt = 3000

# Create x array
x = Lx * np.arange(Nx) / Nx

# Initial condition
u = np.cos(x) + 0.1 * np.cos(x / 16) * (1 + 2 * np.sin(x / 16))

# Call the ksintegrateNaive function (assuming it has been defined)
U, x, t = ksintegrateNaive(u, Lx, dt, Nt, nplot)


# Computing
class ReservoirComputing:
    def __init__(
        self,
        num_neuron=3000,
        spectral_radius=0.4,
        sigma=0.5,
        sparsity=0.98,
        beta=0.0001,
    ):
        self.num_neuron = num_neuron
        self.spectral_radius = spectral_radius
        self.sigma = sigma
        self.beta = beta
        self.Win = np.random.uniform(-sigma, sigma, (num_neuron, 1024))
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
        tind = 0
        print("Training")
        for t in range(steps):
            if t > tind * steps:
                print("=", end="", flush=True)  # Display progress
                tind += 0.01
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

        return self.model.predict(state_history)

    def prediction(self, steps):
        prediction = np.zeros((steps, 1024))
        state = np.zeros((steps, 3000))
        tind = 0
        print("Predicting")
        for t in range(steps):
            if t > tind * steps:
                print("=", end="", flush=True)  # Display progress
                tind += 0.01
            self.state = np.tanh(
                np.dot(self.recurrent_matrix, self.state)
                + np.dot(self.Win, self.model.predict(self.state[None, :])[0])
            )
            prediction[t, :] = self.model.predict(self.state[None, :])[0]
            state[t, :] = self.state

        return prediction, state


data_train, data_test = U[:400], U[400:]
time_train, time_test = t[:400], t[400:]
RC = ReservoirComputing()
reservoir_states = RC.train(data_train)
prediction, state = RC.prediction(data_test.shape[0])

combined_reservoir_states = np.hstack(
    (reservoir_states.transpose(), prediction.transpose())
)
# Plot the result using Matplotlib
plt.figure(figsize=(16, 7))
plt.subplot(3, 1, 1)
plt.imshow(
    U.transpose(),
    aspect="auto",
    origin="lower",
    cmap="jet",
    vmin=-2,
    vmax=2,
    extent=[0, t[-1], 0, x[-1]],
)
plt.xlim([t[0], t[-1]])
plt.ylim([x[0], x[-1]])
plt.xlabel("t")
plt.ylabel("x")
plt.colorbar()

plt.subplot(3, 1, 2)
plt.imshow(
    combined_reservoir_states,
    aspect="auto",
    origin="lower",
    cmap="jet",
    vmin=-2,
    vmax=2,
    extent=[0, t[-1], 0, x[-1]],
)
plt.xlim([t[0], t[-1]])
plt.ylim([x[0], x[-1]])
plt.xlabel("t")
plt.ylabel("x")
plt.colorbar()

plt.subplot(3, 1, 3)
plt.imshow(
    U.transpose() - combined_reservoir_states,
    aspect="auto",
    origin="lower",
    cmap="jet",
    vmin=-2,
    vmax=2,
    extent=[0, t[-1], 0, x[-1]],
)
plt.xlim([t[0], t[-1]])
plt.ylim([x[0], x[-1]])
plt.xlabel("t")
plt.ylabel("x")
plt.colorbar()

plt.show()
