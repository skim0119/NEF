import numpy as np
from lyapynov import DiscreteDS, LCE, ContinuousDS
from sklearn.linear_model import Ridge
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# Define the Reservoir Computing class with Lyapunov Exponent calculation capability
class ReservoirComputingWithLE:
    def __init__(
        self, num_neuron=300, spectral_radius=1.2, sigma=0.1, sparsity=0.98, beta=0.0
    ):
        # Initialize the parameters of the Reservoir Computing system
        self.num_neuron = num_neuron
        self.spectral_radius = spectral_radius
        self.sigma = sigma
        self.beta = beta
        self.Win = np.random.uniform(
            -sigma, sigma, (num_neuron, 3)
        )  # Input weight matrix
        self.recurrent_matrix = (
            np.random.rand(num_neuron, num_neuron) - 0.5
        )  # Reservoir recurrent weight matrix
        self.recurrent_matrix[
            np.random.rand(*self.recurrent_matrix.shape) < sparsity
        ] = 0  # Apply sparsity
        rho_A = max(
            abs(np.linalg.eigvals(self.recurrent_matrix))
        )  # Compute spectral radius
        self.recurrent_matrix *= spectral_radius / rho_A  # Adjust spectral radius
        self.state = np.zeros(self.num_neuron)  # Initialize reservoir state

    def train(self, inputs):
        # Train the Reservoir Computing model
        steps = inputs.shape[0]
        state_history = np.zeros((steps, self.num_neuron))
        for t in range(steps):
            # Update the reservoir state using the input
            self.state = np.tanh(
                np.dot(self.recurrent_matrix, self.state) + np.dot(self.Win, inputs[t])
            )
            state_history[t, :] = self.state

        # Train the output layer using Ridge regression
        self.model = Ridge(alpha=self.beta, fit_intercept=False)
        self.model.fit(state_history[:-1], inputs[1:])
        training_loss = np.mean(
            (inputs[1:] - self.model.predict(state_history[:-1])) ** 2
        )
        print(f"Training Loss (MSE): {training_loss}")
        return state_history, self.model.predict(state_history)

    def prediction(self, steps, initial_input):
        # Make predictions with the trained model
        prediction = np.zeros((steps, 3))
        self.state = np.tanh(
            np.dot(self.recurrent_matrix, self.state) + np.dot(self.Win, initial_input)
        )

        for t in range(steps):
            # Predict the next state using the reservoir dynamics
            self.state = np.tanh(
                np.dot(self.recurrent_matrix, self.state)
                + np.dot(self.Win, self.model.predict(self.state[None, :])[0])
            )
            prediction[t, :] = self.model.predict(self.state[None, :])[0]

        return prediction


# Define the discrete update function of the RC system
def rc_discrete_f(x, t):
    # Update the state using the reservoir recurrent weights and input weights
    state = np.tanh(
        np.dot(RC_LE.recurrent_matrix, x)
        + np.dot(RC_LE.Win, RC_LE.model.predict(x.reshape(1, -1))[0])
    )
    return state


# Define the Jacobian matrix of the RC system
def rc_discrete_jac(x, t):
    # Compute the derivative of the tanh activation function
    tanh_derivative = (
        1
        - (
            np.tanh(
                np.dot(RC_LE.recurrent_matrix, x)
                + np.dot(RC_LE.Win, RC_LE.model.predict(x.reshape(1, -1))[0])
            )
        )
        ** 2
    )
    # Calculate the Jacobian matrix
    jacobian_matrix = tanh_derivative[:, None] * RC_LE.recurrent_matrix
    return jacobian_matrix


# Simulation parameters
T_train = 5000  # Training time steps
T_pred = 1250  # Prediction time steps
dt = 0.02  # Time step size
initial_state = np.array([1.0, 1.0, 1.0])  # Initial state of the Lorenz system
time = np.arange(0, T_train + T_pred) * dt
time_train, time_test = time[:T_train], time[T_train:]


# Define the Lorenz system dynamics
def lorenz_system(state, t):
    x, y, z = state
    a = 10
    b = 28
    c = 8.0 / 3.0
    return [a * (y - x), x * (b - z) - y, x * y - c * z]


# Generate training and testing data using the Lorenz system
data = odeint(lorenz_system, initial_state, time)
data_train, data_test = data[:T_train], data[T_train:]

# Initialize and train the RC model
RC_LE = ReservoirComputingWithLE()
origin_state = RC_LE.state

training_states, reservoir_states = RC_LE.train(data_train)

# Perform prediction using the trained RC model
initial_input = data_train[
    -1
]  # Use the last state from training data as the initial input for prediction
prediction = RC_LE.prediction(data_test.shape[0], initial_input)

# Create a ContinuousDS object for the RC system using the lyapynov library
RC_DS = ContinuousDS(origin_state, 0, rc_discrete_f, rc_discrete_jac, dt)
RC_DS.forward(10**4, False)

# Calculate the Lyapunov exponents using the lyapynov library
lce_values, history = LCE(
    RC_DS, 3, 0, 10**4, True
)  # Use 10^4 steps to compute LCE for more accurate results
print("Lyapunov Exponents from lyapynov library:", lce_values)
