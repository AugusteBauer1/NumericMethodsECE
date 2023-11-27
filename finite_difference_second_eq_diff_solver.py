import numpy as np
import matplotlib.pyplot as plt
import time as t

from direct_methods_linear_systems import LU_solve, resolcholesky

INTERVAL = 20

def force(x):
    """ Define the force function applied on the cable. """
    return x * (1 - x)

def finite_difference_solver(interval, force_function, method='LU'):
    """
    Solve the second order differential equation using the finite difference method.

    Parameters:
    - interval: number of intervals to divide the domain [0, 1]
    - force_function: function representing the force applied on the cable
    - method: the direct method to solve the system ('LU' or 'Cholesky')

    Returns:
    - x: array of positions where displacement is calculated
    - u: array of displacements at the positions x
    """
    
    h = 1 / interval
    x = np.linspace(0, 1, interval + 1)[1:-1]
    b = force_function(x)

    # Generate the coefficient matrix
    A = (np.diag(-np.ones(interval - 2), -1) +
         np.diag(-np.ones(interval - 2), 1) +
         np.diag(2 * np.ones(interval - 1))) * interval**2

    # Choose the method for solving the system
    if method == 'LU':
        start_time = t.time()
        u = LU_solve(A, b)
        end_time = t.time()
    elif method == 'Cholesky':
        start_time = t.time()
        u = resolcholesky(A, b)
        end_time = t.time()
    else:
        raise ValueError("Unknown method. Please use 'LU' or 'Cholesky'.")

    print(f"Execution time with {method} decomposition:", end_time - start_time)
    
    return x, u

def plot_displacement(x, u):
    """Plot the displacement of the cable at the nodes."""

    plt.plot(x, u, marker='o')
    plt.xlabel('Position (x)')
    plt.ylabel('Displacement (u)')
    plt.title('Displacement of the Cable at the Nodes')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    x_LU, u_LU = finite_difference_solver(INTERVAL, force, 'LU')
    plot_displacement(x_LU, u_LU)

    x_Cholesky, u_Cholesky = finite_difference_solver(INTERVAL, force, 'Cholesky')
    plot_displacement(x_Cholesky, u_Cholesky)
