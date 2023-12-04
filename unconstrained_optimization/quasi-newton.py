import numpy as np

from utils import rosenbrock, rosenbrock_prime, _arrow3D
from line_search import wolfe_search

def DFP(B_k, s_k, y_k):
    """
    Update the approximation of the Hessian matrix using the DFP formula.
    """
    s_k = s_k.reshape(-1, 1)
    y_k = y_k.reshape(-1, 1)

    term1 = np.dot(s_k, s_k.T) / np.dot(s_k.T, y_k)
    term2 = np.dot(np.dot(B_k, y_k), np.dot(y_k.T, B_k)) / np.dot(np.dot(y_k.T, B_k), y_k)

    return B_k + term1 - term2

def BFGS(B_k, s_k, y_k):
    """
    Update the approximation of the Hessian matrix using the BFGS formula.
    """
    s_k = s_k.reshape(-1, 1)
    y_k = y_k.reshape(-1, 1)

    term1 = np.dot(y_k, y_k.T) / np.dot(y_k.T, s_k)
    term2 = np.dot(np.dot(B_k, s_k), np.dot(s_k.T, B_k)) / np.dot(np.dot(s_k.T, B_k), s_k)

    return B_k + term1 - term2

def quasi_newton(x_init, f, f_prime, verbose=False, max_iteration=50000, tolerance=1e-5, method='DFP'):
    """
    Minimize a function using the quasi-Newton method. 
    We use the BFGS or DFP formula to update the Hessian approximation.
    """

    # Initialisation
    x = np.array(x_init)
    iterates = [x_init]
    n = len(x)
    B = np.eye(n)
    
    # Boucle principale

    while len(iterates) < max_iteration or np.linalg.norm(x - iterates[-1]) > tolerance:
        # Calcul de la direction de descente
        d = - np.dot(B, f_prime(x))
        
        # Calcul du pas de descente
        alpha = wolfe_search(f, f_prime, x, d)
        
        # Mise à jour de x
        x_new = x + alpha * d
        
        # Mise à jour de B
        s = x_new - x
        y = np.array(f_prime(x_new)) - np.array(f_prime(x))
        
        if method == 'BFGS':
            B = BFGS(B, s, y)
        elif method == 'DFP':
            B = DFP(B, s, y)
        else:
            raise ValueError('The method must be either BFGS or DFP')
        
        # Mise à jour de l'itération
        x = x_new
        iterates.append(x)
        
        if verbose:
            print(f"Iteration {len(iterates) - 1}: {x}")

    return iterates, x

if __name__ == '__main__':
    # test for rosenbrock function in dimension 2
    iterates, x = quasi_newton([-1.8, 1.7], rosenbrock, rosenbrock_prime)
    print(f"Root: {x}")
    print("Number of iterations: ", len(iterates) - 1)
