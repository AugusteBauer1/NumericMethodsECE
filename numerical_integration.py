import numpy as np

def trapezoidal_method(f, a, b, n):
    """ Compute the integral of a function using the trapezoidal rule. """
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    s = y[0] + y[-1] + 2 * np.sum(y[1:-1])
    return h * s / 2

def simpson_method(f, a, b, n):
    """ Compute the integral of a function using Simpson's rule (Precisly Simpson's 1/3 Rule). """
    h = (b - a) / n
    S = 0
    for i in range(n):
        S += f(a + i*h) + 4*f(a + (i + 0.5)*h) + f(a + (i + 1)*h)
    return S * h / 6

def sine(x):
    return np.sin(x)

def gaussian(x):
    return np.exp(-(x**2))

if __name__ == '__main__':

    # Computing integrals using the trapezoidal method
    print(trapezoidal_method(sine, 0, np.pi, 1000))
    print(trapezoidal_method(gaussian, 0, 7, 10))
    print(trapezoidal_method(gaussian, 0, 7, 100))
    print(trapezoidal_method(gaussian, 0, 7, 1000))

    # Computing integrals using Simpson's method
    print(simpson_method(sine, 0, np.pi, 1000))
    print(simpson_method(gaussian, 0, 7, 10))
    print(simpson_method(gaussian, 0, 7, 100))
    print(simpson_method(gaussian, 0, 7, 1000))
