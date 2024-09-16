import sympy as sp
import numpy as np


def tanh_deriv(d, n):
    """
    Compute the first `n` derivatives of the tanh function evaluated at `d`.

    Parameters:
    d: Vector of input values (can be a numpy array or a list)
    n: Number of derivatives to compute

    Returns:
    D: Matrix of derivatives, where each column is the derivative of increasing order
    """
    # Define symbolic variable z
    z = sp.Symbol('z', real=True)

    # Initialize the tanh function
    D = [sp.tanh(z)]

    # Compute the derivatives up to order n
    for i in range(1, n):
        D.append(sp.diff(D[i - 1], z))  # Compute the i-th derivative of tanh

    # Convert the symbolic expressions into numerical functions
    Df = [sp.lambdify(z, deriv) for deriv in D]

    # Evaluate the derivatives at the input values `d`
    if isinstance(d, (list, np.ndarray)):
        D_evaluated = np.array([[Df[i](val) for i in range(n)] for val in d])
    else:
        D_evaluated = np.array([Df[i](d) for i in range(n)])

    return D_evaluated
