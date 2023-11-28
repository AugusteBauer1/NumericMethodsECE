import numpy as np
import matplotlib.pyplot as plt

PRECISION = 20
PRECISION2 =  5000 # Fine-grained precision for interpolation

def lagrange(x, y, x_star):
    """ Compute the interpolated value using Lagrange's interpolation formula. """

    n = len(x)
    y_star = 0
    for i in range(n):
        L = 1
        for j in range(n):
            if j != i:
                L = L * (x_star - x[j]) / (x[i] - x[j])
        y_star = y_star + L * y[i]
    return y_star

def subdivision(a, b, n):
    """ Divide the interval [a, b] into n evenly spaced points. """

    x = np.linspace(a, b, n)
    return x

def tchebychev_subdivision(a, b, n):
    """ Generate n Chebyshev nodes in the interval [a, b]. """

    tchebychev_points = []    
    for i in range(1, n + 1):
        cos_value = np.cos((2 * i - 1) * np.pi / (2 * n))
        x_value = 0.5 * (a + b) + 0.5 * (b - a) * cos_value
        tchebychev_points.append(x_value)

    return np.array(tchebychev_points)

def sinus(x):
    return np.sin(x)

def function(x):
    return 1 / (1 + x**2)

if __name__ == '__main__':
    # Interpolation of the sinus function
    x = subdivision(-np.pi, np.pi, PRECISION)
    y = sinus(x)
    x_interp = np.linspace(-np.pi, np.pi, PRECISION2)
    y_interp = np.array([lagrange(x, y, xi) for xi in x_interp])
    
    plt.plot(x, y, 'bo', label='Original Points')
    plt.plot(x_interp, y_interp, 'r-', label='Lagrange Interpolation')
    plt.legend()
    plt.show()

    # Visualization of the Runge phenomenon
    x = subdivision(-np.pi, np.pi, PRECISION)
    y = function(x)
    y_interp = np.array([lagrange(x, y, xi) for xi in x_interp])

    plt.plot(x, y, 'bo', label='Original Points')
    plt.plot(x_interp, y_interp, 'r-', label='Lagrange Interpolation')
    plt.legend()
    plt.show()

    # Interpolation using Chebyshev nodes
    x_cheb = tchebychev_subdivision(-np.pi, np.pi, PRECISION)
    y_cheb = function(x_cheb)
    y_interp_cheb = np.array([lagrange(x_cheb, y_cheb, xi) for xi in x_interp])

    plt.plot(x_cheb, y_cheb, 'bo', label='Tchebychev Points')
    plt.plot(x_interp, y_interp_cheb, 'r-', label='Lagrange Interpolation with Tchebychev Points')
    plt.legend()
    plt.show()