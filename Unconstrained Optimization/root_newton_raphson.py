import math as m
def f(x):
    return (x-1)*(x**2 +1)

def f_prime(x):
    return 3*x**2 - 2*x +1

def newton_raphson(epsilon, x_init):
    error = 2 * epsilon
    while error > epsilon:
        x_new = x_init - (f(x_init)/f_prime(x_init))
        error = abs(x_new - x_init)
        x_init = x_new

    return x_new

if __name__ == '__main__':
    print(newton_raphson(0.00001,-2))
