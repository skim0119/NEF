import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# 高通滤波器设计函数
fs = 1000
# 生成三个不同截止频率的高通滤波器
of = np.array(
    [
        signal.firwin(3, 0.1, pass_zero=False),  # 高通滤波器
        signal.firwin(3, 0.2, pass_zero=False),
        signal.firwin(3, 0.3, pass_zero=False),
    ]
)

of = np.array(
    (
        [-0.0896879756980144, 0.820624048603971, -0.0896879756980144],
        [-0.159341287490797, 0.681317425018406, -0.159341287490797],
        [-0.211942742334250, 0.576114515331499, -0.211942742334250],
    )
)
# 输出滤波器系数
print("Filter coefficients:")
print(of)

# 可视化滤波器频率响应
for i, coef in enumerate(of):
    w, h = signal.freqz(coef)
    plt.plot(w / np.pi, 20 * np.log10(abs(h)), label=f"High-pass filter {i+1}")

plt.title("High-pass FIR filter frequency response")
plt.xlabel("Normalized Frequency (π rad/sample)")
plt.ylabel("Magnitude (dB)")
plt.legend()
plt.grid()
plt.show()
