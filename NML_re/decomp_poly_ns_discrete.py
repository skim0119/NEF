import numpy as np
from scipy.linalg import inv
from scipy.special import factorial
from tanh_deriv import tanh_deriv

def decomp_poly_ns_discrete(A, B, rs, dv, o, T):
    """
     Decompose reservoir parameters into polynomial basis.

     Parameters:
     A (numpy.ndarray): N x N matrix of the connectivity between N neurons
     B (numpy.ndarray): N x k matrix from the k independent inputs
     rs (numpy.ndarray): N x 1 vector for the equilibrium point of the RNN
     dv (numpy.ndarray): N x 1 vector of the effective bias, A*rs + B*xs + d
     o (int): Scalar for the order of the Taylor series in x
     T (int): Scalar for the number of time points to look back

     Returns:
     Pd1 (numpy.ndarray): p x k matrix of p polynomial basis terms as powers of k inputs
     C1 (numpy.ndarray): N x p matrix of coefficients of the first series expansion
     """
    # Grid indices
    N = A.shape[0]  # Number of neurons
    k = B.shape[1]  # Number of inputs

    # Grid indices
    v = np.eye(k).reshape(k, k)
    Pd1 = np.eye(k)
    v = np.repeat(v[:, np.newaxis, :], repeats=k, axis=1)

    for i in range(2, o):
        Pdp = Pd1 + v
        Pdp = Pdp.reshape(-1, k, order='F')
        Pd1 = np.vstack([Pd1, Pdp])
        _, unique_idx = np.unique(Pd1, axis=0, return_index=True)
        # np.unique 按行来进行字典序排序。因为每一行的数值是不同的，排序时从第一列开始，如果第一列的数字相同则比较第二列
        # return_index=True保留原有的行顺序
        Pd1 = Pd1[np.sort(unique_idx)]

    Pd1 = np.vstack([np.zeros((1, k)), Pd1])
    sI1 = np.argsort(np.max(Pd1, axis=1))
    sI1a = np.argsort(np.sum(Pd1[sI1, :], axis=1))
    sI1 = sI1[sI1a]
    Pd1 = Pd1[sI1, :]

    #% Initial coefficients
    Ars = A @ rs
    # Compute higher order B terms
    Bk = [None] * o
    Bc = [None] * o
    for i in range(1, o):
        PdI = np.where(np.sum(Pd1, axis=1) == i)[0]
        PdI = PdI[:, np.newaxis]
        Bk[i] = np.zeros((N, len(PdI)))
        Bc[i] = np.zeros((1, len(PdI)))
        for j in range(0, len(PdI)):
            Bk[i][:,j] = np.prod(B ** Pd1[PdI[j], :], axis=1)
            Bc[i][:,j] = factorial(i) / np.prod(factorial(Pd1[PdI[j], :]))

        Bk[i - 1] = Bk[i]
        Bc[i - 1] = Bc[i]
        Bk[i] = [[np.zeros((N, len(PdI)))]]
        Bc[i] = np.zeros((1, len(PdI)))

    D = tanh_deriv(dv, o+1)

    Ars = Ars.reshape(-1, 1)  # 确保 Ars 是列向量
    D = np.squeeze(D)
    DD = D[:, :-1] - D[:, 1:] * Ars

    #Sole higher derivative terms
    CM = DD[:, 0].reshape(N, 1)

    for j in range(1, o):
        CM = np.concatenate((CM, (DD[:, j].reshape(N, 1)*(Bc[j-1]*Bk[j-1])/factorial(j))), axis = 1)


    C1 = np.tile(CM[:, :, np.newaxis], (1, 1, T))

    As = (1 - np.tanh(dv) ** 2) * A
    for i in range(1, T):
        C1[:, :, i] = As @ C1[:, :, i-1]


    C1[:, 0, :] = np.tile(rs, (1, 1, T))/T
    C1 = np.flip(C1, axis=2)
    print("Complete")
    return Pd1, C1












