import numpy as np

from utils import rosenbrock, rosenbrock_prime, _arrow3D
from line_search import wolfe_search, golden_section_search



def DFP(s_k, y_k):
    """
    Update the approximation of the Hessian matrix using the DFP formula.
    """
    B_old = np.eye(len(s_k))

    s_k = s_k.reshape(-1, 1)
    y_k = y_k.reshape(-1, 1)

    tol = 1e-6
    error = 2 * tol

    while error > tol:
        B_new = B_old + np.dot(s_k, s_k.T) / np.dot(s_k.T, y_k) - np.dot(np.dot(B_old, y_k), np.dot(y_k.T, B_old)) / np.dot(np.dot(y_k.T, B_old), y_k)
        error = np.linalg.norm(B_new - B_old)
        B_old = B_new


    return B_new

def BFGS(s_k, y_k):
    """
    Update the approximation of the Hessian matrix using the BFGS formula.
    """
    B_old = np.eye(len(s_k))

    s_k = s_k.reshape(-1, 1)
    y_k = y_k.reshape(-1, 1)

    tol = 1e-6
    error = 2 * tol

    while error > tol:
        B_new = B_old + np.dot(y_k, y_k.T) / np.dot(y_k.T, s_k) - np.dot(np.dot(B_old, s_k), np.dot(s_k.T, B_old)) / np.dot(np.dot(s_k.T, B_old), s_k)
        error = np.linalg.norm(B_new - B_old)
        B_old = B_new


    return B_new

def quasi_newton(x_init, f, f_prime, verbose=False, max_iteration=30000, tolerance=1e-5, method='DFP'):
    """
    Minimize a function using the quasi-Newton method.
    We use the BFGS or DFP formula to update the Hessian approximation.
    """

    # Initialisation
    x = np.array(x_init)
    iterates = [x_init]
    n = len(x)
    B = np.eye(n)

    alpha = 0.0005

    
    while len(iterates) < max_iteration or np.linalg.norm(x - iterates[-1]) > tolerance:
        # Compute the search direction
        d = - np.dot(B, f_prime(x))

        x = x +  d * alpha

        s = x - iterates[-1]

        y = np.array(f_prime(x)) - np.array(f_prime(iterates[-1]))
        if method == 'DFP':
            B = DFP(s, y)
        elif method == 'BFGS':
            B = BFGS(s, y)
        else:
            raise ValueError("The method must be 'DFP' or 'BFGS'.")

        # Save the next iterate
        iterates.append(x)



    return iterates, x

if __name__ == '__main__':
    # test for rosenbrock function in dimension 2
    iterates, x = quasi_newton([-1.8, 1.7], rosenbrock, rosenbrock_prime)
    print(f"Root: {x}")
    print("Number of iterations: ", len(iterates) - 1)
