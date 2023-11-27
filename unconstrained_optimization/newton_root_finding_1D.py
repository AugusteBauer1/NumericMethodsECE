import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return (x-1)*(x**2 +1)

def f_prime(x):
    return 3*x**2 - 2*x +1


def newton_raphson_R1(epsilon, x_init):
    """
    Finds the root of a function f in R1 using the Newton-Raphson method.
    Complexity: O(n^3), where n is the number of iterations until convergence.
    """
    x = x_init - f(x_init)/f_prime(x_init)
    iterates = [x_init, x]
    while abs(x - iterates[-2]) > epsilon:
        x = x - f(x)/f_prime(x)
        iterates.append(x)
    return iterates, len(iterates), x

if __name__ == '__main__':
    iterates, n, x = newton_raphson_R1(1e-5, 0.5 + 2j)
    print(f"Number of iterations: {n}")
    print(f"Root: {x}")
    
    # plot the error between the iterates and the root
    plt.plot(np.arange(n), [abs(i - x) for i in iterates])
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Error Between the Iterates and the Root')
    plt.grid(True)
    plt.show()

    real = np.linspace(-2, 2, 1000)
    imag = np.linspace(-2, 2, 1000)
    real, imag = np.meshgrid(real, imag)
    complex_grid = real + 1j * imag

    # plot the complete path of the iterates for each point of the grid
    plt.figure(figsize=(8, 6))

    len_iterates = np.zeros(complex_grid.shape)
    # Apply the Newton-Raphson method to each point of the grid
    for i in range(complex_grid.shape[0]):
        for j in range(complex_grid.shape[1]):
            iterates, len_iterates[i, j], x = newton_raphson_R1(1e-5, complex_grid[i, j])
            if i % 200 == 0 and j % 200 == 0:
                for k in range(int(len_iterates[i,j]) - 1):
                    start = iterates[k]
                    end = iterates[k + 1]
                    plt.arrow(start.real, start.imag, end.real - start.real, end.imag - start.imag, shape='full', lw=1, length_includes_head=True, head_width=0.05)

    

    # Set the limits for the axes
    ax = plt.gca()
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])

    plt.xlabel('Real part')
    plt.ylabel('Imaginary part')
    plt.title('Complete Path of the Iterates for Each Point of the Grid of the Complex Plane')

    # plot the number of iterations until convergence for each point of the grid

    plt.figure(figsize=(8, 6))
    plt.pcolor(real, imag, len_iterates)
    plt.colorbar()
    plt.xlabel('Real part')
    plt.ylabel('Imaginary part')
    plt.title('Number of Iterations Until Convergence for Each Point of the Grid of the Complex Plane')

    plt.show()
    




