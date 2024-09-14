import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
def ksintegrateNaive(u, Lx, dt, Nt, nplot):
    """
    Naive integration of the Kuramoto-Sivashinsky equation:
    u_t = -u*u_x - u_xx - u_xxxx, periodic boundary conditions
    """
    Nx = len(u)                      # number of grid points
    kx = np.concatenate([np.arange(0, Nx//2), [0], np.arange(-Nx//2+1, 0)])  # integer wavenumbers
    alpha = 2 * np.pi * kx / Lx       # real wavenumbers: exp(i*alpha*x)
    D = 1j * alpha                    # D = d/dx operator in Fourier space
    L = alpha**2 - alpha**4           # linear operator: -D^2 - D^4 in Fourier space
    G = -0.5 * D                      # -1/2 D operator in Fourier space
    Nplot = Nt // nplot + 1           # total number of saved time steps

    x = np.linspace(0, Lx, Nx, endpoint=False)  # x values
    t = np.arange(Nplot) * dt * nplot           # time array
    U = np.zeros((Nplot, Nx))                   # to store solution at saved time steps

    # some convenience variables
    dt2 = dt / 2
    dt32 = 3 * dt / 2
    A = 1 + dt2 * L
    B = 1 / (1 - dt2 * L)

    Nn = G * fft(u * u)              # -1/2 d/dx(u^2) = -u u_x (spectral calculation)
    Nn1 = Nn                         # initialize N^{n-1}

    U[0, :] = u                      # save initial condition to matrix U
    np_idx = 1                       # counter for saved data

    u = fft(u)                       # transform u to spectral

    # time-stepping loop
    for n in range(Nt):
        Nn1 = Nn                     # shift N^{n-1} <- N^n
        u_real = np.real(ifft(u))    # convert u back to physical space
        Nn = G * fft(u_real**2)      # compute N^n = -u u_x

        # CNAB2 formula: B * (A * u + dt32 * Nn - dt2 * Nn1)
        u = B * (A * u + dt32 * Nn - dt2 * Nn1)

        if (n + 1) % nplot == 0:
            U[np_idx, :] = np.real(ifft(u))  # save solution at this time step
            np_idx += 1

    return U, x, t

# Parameters
Lx = 128
Nx = 1024
dt = 0.25
nplot = 8
Nt = 5000

# Create x array
x = Lx * np.arange(Nx) / Nx

# Initial condition
u = np.cos(x) + 0.1 * np.cos(x / 16) * (1 + 2 * np.sin(x / 16))

# Call the ksintegrateNaive function (assuming it has been defined)
U, x, t = ksintegrateNaive(u, Lx, dt, Nt, nplot)

# Plot the result using Matplotlib
plt.pcolor(x, t, U, cmap='jet', vmin=-2, vmax=2)
plt.xlim([x[0], x[-1]])
plt.ylim([t[0], t[-1]])
plt.xlabel("x")
plt.ylabel("t")
plt.title("Kuramoto-Sivashinsky dynamics")
plt.colorbar(label="u(x, t)")
plt.show()