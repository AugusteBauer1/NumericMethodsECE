import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))
from linear_systems.direct_methods_linear_systems import LU_solve

def g(x):
    return [x[0]**2 - x[1] - 1, x[1]**2 - x[1]]

def g_jacobi(x):
    return [[2*x[0], -1], [-1, 2*x[1]]]

def newton_raphson_R2(epsilon, x_init):
    """
    Finds the root of a function g in R2 using the Newton-Raphson method.
    Complexity: O(n^3), where n is the number of iterations until convergence.
    """

    # first iteration
    # test if the matrix is invertible
    
    delta = LU_solve(np.array(g_jacobi(x_init)), -np.array(g(x_init)))
    x = x_init + delta
    iterates = [x_init, x]

    # next iterations
    while abs(x[0] - iterates[-2][0]) + abs(x[1] - iterates[-2][1]) > epsilon:
        delta = LU_solve(np.array(g_jacobi(x)), -np.array(g(x)))
        x = x + delta
        iterates.append(x)

    return iterates, len(iterates), x

if __name__ == '__main__':
    iterates, n, x = newton_raphson_R2(1e-5, [-1.4, -1])
    print(f"Number of iterations: {n}")
    print(f"Root: {x}")

    # plot the error between the iterates and the root
    plt.plot(np.arange(n), [abs(i[0] - x[0]) + abs(i[1] - x[1]) for i in iterates])
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Error Between the Iterates and the Root')
    plt.grid(True)
    plt.show()

    x1 = np.linspace(-2, 2, 50)
    x2 = np.linspace(-2, 2, 50)
    x1, x2 = np.meshgrid(x1, x2)
    grid = np.array([x1, x2])

    # plot the complete path of the iterates for each point of the grid
    plt.figure(figsize=(8, 6))

    len_iterates = np.zeros(grid.shape[1:])
    # Apply the Newton-Raphson method to each point of the grid
    for i in range(grid.shape[1]):
        for j in range(grid.shape[2]):
            # test if the matrix is invertible for each point of the grid
            if np.linalg.det(g_jacobi(grid[:, i, j])) != 0:
                iterates,len_iterates[i,j], x = newton_raphson_R2(1e-5, grid[:, i, j])
                if i % 5 == 0 and j % 10 == 0:
                    for k in range(len(iterates) - 1):
                        start = iterates[k]
                        end = iterates[k + 1]
                        plt.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1], shape='full', lw=1, length_includes_head=True, head_width=0.05)

    # Set the limits for the axes
    ax = plt.gca()
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Complete Path of the Iterates for Each Point of the Grid')

    # plot the number of iterations until convergence for each point of the grid

    # Parameter for the circle
    t = np.linspace(0, 2 * np.pi, 100)
    x_circle = np.sqrt(2) * np.cos(t)
    y_circle = np.sqrt(2) * np.sin(t)

    # Parameters for the hyperbolas
    t_hyp = np.linspace(-2 * np.pi, 2 * np.pi, 100)
    x_hyperbola1 = np.cosh(t_hyp)
    y_hyperbola1 = np.sinh(t_hyp)
    x_hyperbola2 = -np.cosh(t_hyp)
    y_hyperbola2 = np.sinh(t_hyp)

    # plot the circle and the hyperbolas
    plt.plot(x_circle, y_circle, color='orange')
    plt.plot(x_hyperbola1, y_hyperbola1, color='blue')
    plt.plot(x_hyperbola2, y_hyperbola2, color='blue')

    # plot the roots
    plt.plot(x[0], x[1], marker='o', color='red')
    plt.plot(-x[0], x[1], marker='o', color='red')
    plt.plot(x[0], -x[1], marker='o', color='red')
    plt.plot(-x[0], -x[1], marker='o', color='red')
    
    plt.figure(figsize=(8, 6))
    plt.contourf(x1, x2, len_iterates)
    plt.colorbar()
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Number of Iterations Until Convergence for Each Point of the Grid')

    plt.show()
    

