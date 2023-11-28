import math as m

def f(x):
    """ Define the function to be solved. """
    return 10 * x - 9 * m.exp(-x)

def phi(x):
    """ Reformulation of f(x) = 0 to x = phi(x) for the fixed point method. """
    return (9 * m.exp(-x)) / 10

def phi_prime(x):
    """ Derivative of phi. """
    return (- 9 * m.exp(-x)) / 10

def bisection_method(func, a, b, epsilon):
    """
    Find the root of a function using the bisection method.
    Complexity: O(log(b-a/epsilon)), assuming func is O(1).
    """
    if func(a) * func(b) > 0:
        print("Bisection method conditions are not satisfied.")
        return None

    while (b - a) / 2.0 > epsilon:
        c = (a + b) / 2.0
        if func(c) == 0:
            return c
        elif func(c) * func(a) < 0:
            b = c
        else:
            a = c

    return (a + b) / 2.0

def fixed_point(func, func_prime, x0, epsilon):  
    """
    Find the root of a function using the fixed point iteration.
    Complexity: Iterative, depends on the convergence criteria (epsilon).
    """
    if abs(func_prime(x0)) >= 1:
        print("The function is not convergent for the given starting point.")
        return None

    x = func(x0)
    error = epsilon*3
    while error > epsilon:
        x0 = x
        x = func(x0)
        error = abs(x-x0)
    return x

if __name__ == '__main__':
    # Test bisection method
    root_bisection = bisection_method(f, 0, 1, 0.0001)
    print(f"Root using Bisection Method: {root_bisection}")

    # Test fixed point iteration
    root_fixed_point = fixed_point(phi, phi_prime, 0, 0.0001)
    print(f"Root using Fixed Point Iteration: {root_fixed_point}")