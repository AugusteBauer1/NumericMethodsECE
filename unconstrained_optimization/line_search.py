import numpy as np


def golden_section_search(f_alpha, a, b, tolerance=1e-5):
    """
    Use the golden section search method to find the optimal step size given a function f.
    We can use the golden section method assuming that the function is one-dimensional (i.e. f: R -> R) and that it is unimodal.
    Note that this is a zero-order method, we do not care about differentiability or even continuity of the function.
    """
    (a, b) = (min(a, b), max(a, b))
    h = b - a
    if h <= tolerance:
        return (a + b)/2

    inv_golden_ratio = (np.sqrt(5) - 1) / 2
    inv_golden_ratio_sq = (3 - np.sqrt(5)) / 2

    x1 = a + inv_golden_ratio_sq * h
    x2 = a + inv_golden_ratio * h
    v1 = f_alpha(x1)
    v2 = f_alpha(x2)
    
    while np.abs(b - a) > tolerance:

        if v1 < v2:
            b = x2
            x2 = x1
            v2 = v1
            x1 = a + b - x2
            v1 = f_alpha(x1)

        else:
            a = x1
            x1 = x2
            v1 = v2
            x2 = a + b - x1
            v2 = f_alpha(x2)

    return (a + b) / 2

def wolfe_search(f_alpha, f_alpha_prime, x, d, rho=0.5, c1=1e-3, c2=0.90, tolerance=0.0001):
    """
    Use the Wolfe search method to find a good step size given a function f.
    This method is called an inexact line search, it does not guarantee optimality but a sufficient decrease in the function.
    """
    rho_minus, rho_plus = 0, float('inf')

    while True:
        x_new = x + rho * d
        f_new = f_alpha(x_new)
        f_old = f_alpha(x)
        grad_f_old = f_alpha_prime(x)

        if f_new > f_old + c1 * rho * np.dot(grad_f_old, d):
            rho_plus = rho
        elif np.dot(f_alpha_prime(x_new), d) < c2 * np.dot(grad_f_old, d):
            rho_minus = rho
        else:
            return rho

        if rho_plus < float('inf'):
            if rho_plus - rho_minus < tolerance:
                return rho
            rho = (rho_plus + rho_minus) / 2
        else:
            rho = 2 * rho_minus