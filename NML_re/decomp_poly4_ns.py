import numpy as np
from scipy.linalg import inv
from scipy.special import factorial
from tanh_deriv import tanh_deriv

# 假设 tanh_deriv 是已定义的函数，按你的说明，不需要重写
# def tanh_deriv(dv, o):
#     """Placeholder for tanh_deriv function"""
#     pass


def decomp_poly4_ns(A, B, rs, dv, gam, o):
    """
    Function to decompose reservoir parameters into polynomial basis.

    Parameters:
    A:        N x N matrix of the connectivity between N neurons
    B:        N x k matrix from the k independent inputs
    rs:       N x 1 vector for the equilibrium point of the RNN
    dv:       N x 1 vector of the effective bias, A*rs + B*xs + d
    gam:      scalar for the time constant of the RNN
    o:        scalar for the order of the Taylor series in x

    Returns:
    Pd1:      p x k matrix of p polynomial basis terms as powers of k inputs
    C1, C2, C3a, C3b, C4a, C4b, C4c: coefficients of the series expansion
    """

    # Base parameters
    N = A.shape[0]  # Number of neurons
    k = B.shape[1]  # Number of inputs

    # Grid indices
    v = np.eye(k).reshape(k, k)
    Pd1 = np.eye(k)
    v = np.repeat(v[:, np.newaxis, :], repeats=k, axis=1)

    for i in range(2, o):
        Pdp = Pd1 + v
        Pdp = Pdp.reshape(-1, 3, order="F")
        Pd1 = np.vstack([Pd1, Pdp])
        _, unique_idx = np.unique(Pd1, axis=0, return_index=True)
        Pd1 = Pd1[np.sort(unique_idx)]

    Pd1 = np.vstack([np.zeros((1, k)), Pd1])
    sI1 = np.argsort(np.max(Pd1, axis=1))
    sI1a = np.argsort(np.sum(Pd1[sI1, :], axis=1))
    sI1 = sI1[sI1a]
    Pd1 = Pd1[sI1, :]

    # Coefficients
    Ars = A @ rs

    # Compute higher order B terms
    Bk = [None] * o
    Bc = [None] * o
    for i in range(1, o):
        PdI = np.where(np.sum(Pd1, axis=1) == i)[0]
        PdI = PdI[:, np.newaxis]
        Bk[i] = np.zeros((N, len(PdI)))
        Bc[i] = np.zeros((1, len(PdI)))  # 创建 (1, len(PdI)) 的二维数组
        for j in range(len(PdI)):
            Bk[i][:, j] = np.prod(B ** Pd1[PdI[j], :], axis=1)
            Bc[i][0, j] = factorial(i) / np.prod(
                factorial(Pd1[PdI[j], :])
            )  # 保持 Bc 是二维数组

        Bk[i - 1] = Bk[i]
        Bc[i - 1] = Bc[i]
        Bk[i] = [[np.zeros((N, len(PdI)))]]
        Bc[i] = np.zeros((1, len(PdI)))

        # Compute tanh derivatives
    D = tanh_deriv(dv, o + 4)  # 假设你有自己的 tanh_deriv 函数
    Ars = Ars.reshape(-1, 1)  # 确保 Ars 是列向量
    D = np.squeeze(D)

    # 确保 D 和 Ars 形状兼容并进行广播
    DD = D[:, 1:] * Ars - D[:, :-1]
    # Prefactors
    As = (1 - np.tanh(dv) ** 2) * A - np.eye(N)
    AsI = inv(As)
    AsI2 = AsI @ AsI
    AsI3 = AsI2 @ AsI
    AsI4 = AsI3 @ AsI

    # Sole higher derivative terms
    CM = DD[:, :4].reshape(N, 1, 4)

    for j in range(1, o):
        # print((Bc[j - 1] * Bk[j - 1]).shape
        CM = np.concatenate(
            (
                CM,
                (DD[:, j : j + 4].reshape(N, 1, 4))
                * ((Bc[j - 1] * Bk[j - 1])[:, :, np.newaxis] / factorial(j - 1)),
            ),
            axis=1,
        )
        # print(CM.shape)
    # xdot terms

    C1 = np.linalg.solve(As, CM[:, :, 0])

    CM_1 = CM[:, :, 1]
    CM_1 = CM_1[:, :, np.newaxis]
    C2 = np.zeros((N, 10, o))  # need to be changed
    for i in range(0, o):
        C2[:, :, i] = (
            AsI2 @ (CM_1 * (Bc[0] * Bk[0]).reshape(N, 1, Bc[0].shape[1]) / gam)[:, :, i]
        )

    C3b = np.zeros((N, 10, o))
    for i in range(o):
        C3b[:, :, i] = AsI @ C2[:, :, i] / gam

    # 计算 C4c
    C4c = np.zeros((N, 10, o))
    for i in range(o):
        C4c[:, :, i] = AsI @ C3b[:, :, i] / gam

    # 计算 xdot^2 terms - C3a
    CM_2 = CM[:, :, 2]
    CM_2 = CM_2[:, :, np.newaxis]
    C3a = np.zeros((N, 10, o))
    for i in range(o):
        C3a[:, :, i] = (
            AsI3
            @ (CM_2 * (Bc[1] * Bk[1]).reshape(N, 1, Bc[1].shape[1]) / gam**2)[:, :, i]
        )

    # 计算 xdot^2 terms - C4b
    C4b = np.zeros((N, 10, o))
    for i in range(o):
        C4b[:, :, i] = (
            3 * AsI4 @ (CM_2 * Bk[1].reshape(N, 1, Bc[1].shape[1]) / gam**3)[:, :, i]
        )

    # 计算 xdot^3 terms - C4a
    CM_3 = CM[:, :, 3]
    CM_3 = CM_3[:, :, np.newaxis]
    C4a = np.zeros((N, 10, o))
    for i in range(o):
        C4a[:, :, i] = (
            AsI4
            @ (CM_3 * (Bc[2] * Bk[2]).reshape(N, 1, Bc[2].shape[1]) / gam**3)[:, :, i]
        )

    print("Complete")

    return Pd1, C1, C2, C3a, C3b, C4a, C4b, C4c
