import numpy as np
from scipy.sparse.linalg import svds


class ReservoirTanhB:
    def __init__(self, A, B, rs, xs, delT, gam):
        """
        Constructor to initialize the reservoir.
        """
        # Matrices
        self.A = A
        self.B = B
        # States and fixed points
        self.rs = rs
        self.xs = xs
        self.d = np.arctanh(rs) - A @ rs - B @ xs
        # Time
        self.delT = delT
        self.gam = gam
        # Initialize reservoir states to zero
        self.r = np.zeros(A.shape[0])

    def train(self, x):
        """
        Train the reservoir using input x.
        """
        nInd = 0
        print("." * 100)  # Progress bar
        nx = x.shape[1]  # Number of time steps
        D = np.zeros((self.A.shape[0], nx))
        D[:, 0] = self.r.reshape(-1)  # Store the initial state
        for i in range(1, nx):
            if i > nInd * nx:
                print("=", end="", flush=True)  # Display progress
                nInd += 0.01

            self.propagate(x[:, i - 1, :])  # Propagate states
            D[:, i] = self.r.reshape(-1)
        return D

    def trainSVD(self, x, k):
        """
        Train the reservoir using input x with SVD.
        """
        U, S, Vt = svds(self.A, k)
        self.US = U @ np.diag(S)
        self.V = Vt.T

        nx = x.shape[1]
        D = np.zeros((self.A.shape[0], nx))
        D[:, 0] = self.r
        for i in range(1, nx):
            self.propagateSVD(x[:, i - 1, :])
            D[:, i] = self.r
        return D

    def train4(self, x):
        """
        Train the reservoir using input x with 4 RK4 steps.
        """
        nx = x.shape[1]
        D = np.zeros((self.A.shape[0], nx, 4))
        D[:, 0, 0] = self.r
        for i in range(1, nx):
            self.propagate(x[:, i - 1, :])
            D[:, i] = self.r
            D[:, i - 1, 1:4] = (
                D[:, i - 1, 0] + np.array([self.k1 / 2, self.k2 / 2, self.k3]).T
            )
        return D

    def predict(self, W, nc):
        """
        Predict with feedback W for nc time steps.
        """
        self.R = self.A + self.B @ W  # Feedback matrix
        self.W = W
        D = np.zeros((self.R.shape[0], nc))
        D[:, 0] = self.r
        for i in range(1, nc):
            self.propagate_x()  # Propagate with feedback
            D[:, i] = self.r
        return D

    def predictSVD(self, W, nc, k):
        """
        Predict with feedback W for nc time steps with SVD.
        """
        U, S, Vt = svds(self.A + self.B @ W, k)
        self.US = U @ np.diag(S)
        self.V = Vt.T

        D = np.zeros((self.A.shape[0], nc))
        D[:, 0] = self.r
        for i in range(1, nc):
            self.propagateSVD_x()  # Propagate with SVD and feedback
            D[:, i] = self.r
        return D

    def predict4(self, W, nc):
        """
        Predict with feedback W for nc time steps, storing RK4 steps.
        """
        self.R = self.A + self.B @ W
        self.W = W
        D = np.zeros((self.R.shape[0], nc, 4))
        D[:, 0, 0] = self.r
        for i in range(1, nc):
            self.propagate_x()
            D[:, i] = self.r
            D[:, i - 1, 1:4] = (
                D[:, i - 1, 0] + np.array([self.k1 / 2, self.k2 / 2, self.k3]).T
            )
        return D

    # Runge-Kutta 4th order integration
    def propagate(self, x):
        """
        Propagate the states using RK4 for driven reservoir.
        """
        x = x[:, np.newaxis, :]
        self.k1 = self.delT * self.del_r(self.r, x[:, 0, 0])
        self.k2 = self.delT * self.del_r(self.r + self.k1 / 2, x[:, 0, 1])
        self.k3 = self.delT * self.del_r(self.r + self.k2 / 2, x[:, 0, 2])
        self.k4 = self.delT * self.del_r(self.r + self.k3, x[:, 0, 3])

        self.r = self.r + (self.k1 + 2 * self.k2 + 2 * self.k3 + self.k4) / 6

    def propagate_x(self):
        """
        Propagate the states with feedback using RK4.
        """
        self.k1 = self.delT * self.del_r_x(self.r)
        self.k2 = self.delT * self.del_r_x(self.r + self.k1 / 2)
        self.k3 = self.delT * self.del_r_x(self.r + self.k2 / 2)
        self.k4 = self.delT * self.del_r_x(self.r + self.k3)
        self.r = self.r + (self.k1 + 2 * self.k2 + 2 * self.k3 + self.k4) / 6

    def propagateSVD(self, x):
        """
        Propagate the states using RK4 for driven reservoir with SVD.
        """
        self.k1 = self.delT * self.delSVD_r(self.r, x[:, 0, 0])
        self.k2 = self.delT * self.delSVD_r(self.r + self.k1 / 2, x[:, 0, 1])
        self.k3 = self.delT * self.delSVD_r(self.r + self.k2 / 2, x[:, 0, 2])
        self.k4 = self.delT * self.delSVD_r(self.r + self.k3, x[:, 0, 3])
        self.r = self.r + (self.k1 + 2 * self.k2 + 2 * self.k3 + self.k4) / 6

    def propagateSVD_x(self):
        """
        Propagate the states with feedback using RK4 with SVD.
        """
        self.k1 = self.delT * self.delSVD_r_x(self.r)
        self.k2 = self.delT * self.delSVD_r_x(self.r + self.k1 / 2)
        self.k3 = self.delT * self.delSVD_r_x(self.r + self.k2 / 2)
        self.k4 = self.delT * self.delSVD_r_x(self.r + self.k3)
        self.r = self.r + (self.k1 + 2 * self.k2 + 2 * self.k3 + self.k4) / 6

    # ODE definitions
    def del_r(self, r, x):
        """
        Define the ODE for driven reservoir.
        """

        return self.gam * (
            -r + np.tanh(self.A @ r + (self.B @ x)[:, np.newaxis] + self.d)
        )

    def del_r_x(self, r):
        """
        Define the ODE for feedback reservoir.
        """
        return self.gam * (-r + np.tanh(self.A @ r + self.B @ (self.W @ r) + self.d))

    def delSVD_r(self, r, x):
        """
        Define the ODE for driven reservoir with SVD.
        """
        return self.gam * (-r + np.tanh(self.US @ (self.V @ r) + self.B @ x + self.d))

    def delSVD_r_x(self, r):
        """
        Define the ODE for feedback reservoir with SVD.
        """
        return self.gam * (-r + np.tanh(self.US @ (self.V @ r) + self.d))
