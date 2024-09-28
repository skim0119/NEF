import numpy as np
from lyapynov import DiscreteDS, LCE, ContinuousDS
from sklearn.linear_model import Ridge
from scipy.integrate import odeint
import matplotlib.pyplot as plt
# 定义带有Lyapunov指数计算能力的 Reservoir Computing 类
class ReservoirComputingWithLE:
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
        training_loss = np.mean((inputs[1:] - self.model.predict(state_history[:-1]))**2)
        print(f"Training Loss (MSE): {training_loss}")
        return state_history, self.model.predict(state_history)

    def prediction(self, steps, initial_input):
        prediction = np.zeros((steps, 3))
        self.state = np.tanh(np.dot(self.recurrent_matrix, self.state) + np.dot(self.Win, initial_input))

        for t in range(steps):
            self.state = np.tanh(np.dot(self.recurrent_matrix, self.state) + np.dot(self.Win, self.model.predict(self.state[None, :])[0]))
            prediction[t, :] = self.model.predict(self.state[None, :])[0]

        return prediction

# 定义RC系统的离散更新函数
def rc_discrete_f(x, t):
    state = np.tanh(np.dot(RC_LE.recurrent_matrix, x) + np.dot(RC_LE.Win, RC_LE.model.predict(x.reshape(1, -1))[0]))
    return state

# 定义RC系统的雅可比矩阵
def rc_discrete_jac(x, t):
    tanh_derivative = 1 - (np.tanh(np.dot(RC_LE.recurrent_matrix, x) + np.dot(RC_LE.Win, RC_LE.model.predict(x.reshape(1, -1))[0]))) ** 2
    jacobian_matrix = tanh_derivative[:, None] * RC_LE.recurrent_matrix
    return jacobian_matrix

# 仿真参数
T_train = 5000
T_pred = 1250
dt = 0.02
initial_state = np.array([1.0, 1.0, 1.0])
time = np.arange(0, T_train + T_pred) * dt
time_train, time_test = time[:T_train], time[T_train:]

def lorenz_system(state, t):
    x, y, z = state
    a = 10
    b = 28
    c = 8.0 / 3.0
    return [a * (y - x), x * (b - z) - y, x * y - c * z]

data = odeint(lorenz_system, initial_state, time)
data_train, data_test = data[:T_train], data[T_train:]

# 初始化并训练RC模型
RC_LE = ReservoirComputingWithLE()
origin_state = RC_LE.state

training_states, reservoir_states = RC_LE.train(data_train)

# 在预测阶段创建离散系统对象
initial_input = data_train[-1]  # 使用训练结束时的最后一个状态作为预测的初始输入
prediction = RC_LE.prediction(data_test.shape[0], initial_input)

# 创建离散系统对象
RC_DS = ContinuousDS(origin_state, 0, rc_discrete_f, rc_discrete_jac, dt)
RC_DS.forward(10**4, False)
# 使用lyapynov库计算Lyapunov指数
lce_values, history = LCE(RC_DS, 3, 0, 10**4, True)  # 使用10^4步来计算LCE，结果会更准确
print("Lyapunov Exponents from lyapynov library:", lce_values)
