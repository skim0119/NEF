import numpy as np


class Thomas:
    def __init__(self, x0, delT):
        """
        Constructor to initialize the state and time step.

        Parameters:
        x0: Initial state (3x1 vector)
        delT: Time step for the simulation
        """
        self.x0 = x0  # Initial state
        self.x = np.copy(x0)  # Current state
        self.delT = delT  # Time step

    def del_x(self, x):
        """
        Defines the Thomas attractor's differential equations.

        Parameters:
        x: The current state of the system

        Returns:
        dx: The derivative of x (3x1 vector)
        """
        dx = np.zeros_like(x)
        dx[0] = -3.7 * x[0] + 5 * np.sin(x[1] * 4)
        dx[1] = -3.7 * x[1] + 5 * np.sin(x[2] * 4)
        dx[2] = -3.7 * x[2] + 5 * np.sin(x[0] * 4)
        return dx

    def propagate(self, n):
        """
        Propagates the system for n steps using the Runge-Kutta 4th order method.

        Parameters:
        n: Number of steps to simulate

        Returns:
        X: A 3xn matrix containing the trajectory of the system
        """
        nInd = 0  # Counter for progress display
        X = np.zeros((3, n, 4))  # Store the state at each step
        X[:, 0, 1] = self.x

        print("." * 100)  # Progress bar
        for i in range(1, n):
            if i > nInd * n:
                print('=', end='', flush=True)  # Display progress
                nInd += 0.01

            # Runge-Kutta 4th order method to propagate the system
            k1 = self.delT * self.del_x(self.x)
            k2 = self.delT * self.del_x(self.x + k1 / 2)
            k3 = self.delT * self.del_x(self.x + k2 / 2)
            k4 = self.delT * self.del_x(self.x + k3)
            self.x = self.x + (k1 + 2 * k2 + 2 * k3 + k4) / 6

            # Store the new state
            X[:, i, 0] = self.x
            X[:, i-1, 1:4] = np.stack([self.x+k1/2, self.x+k2/2, self.x+k3], axis = -1)

        print()  # Finish progress bar
        return X

