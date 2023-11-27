import numpy as np
import matplotlib.pyplot as plt

from utils import rosenbrock, rosenbrock_prime, Arrow3D, _arrow3D

def gradient_descent(x_init, learning_rate = 1e-3, tolerance=1e-5, max_iteration=10000):
    """
    Use the gradient descent method with rate to find the minimum of a function f in R2.
    """

    x = x_init
    iterates = [x_init]
    residual = 2 * tolerance

    while np.linalg.norm(residual) > tolerance and len(iterates) < max_iteration:
        delta = - learning_rate * np.array(rosenbrock_prime(x))
        x = x + delta
        residual = np.linalg.norm(x - iterates[-1])
        iterates.append(x)

    return iterates, x


if __name__ == '__main__':
    step_size = 1000
    # test the gradient_descent function with the Rosenbrock function with a learning rate of 1e-3 and an initial point (-1.9, 2)
    iterates, x = gradient_descent([-1.8, 1.7])
    print(f"Root: {x}")
    print("Number of iterations: ", len(iterates) - 1)
    # first print the function around he point (1, 1) in 3D
    x1 = np.linspace(-2, 2, 50)
    x2 = np.linspace(-2, 2, 50)
    x1, x2 = np.meshgrid(x1, x2)
    grid = np.array([x1, x2])
    z = rosenbrock(grid)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x1, x2, z, cmap='viridis', edgecolor='none')
    ax.text2D(-0.27, 0, f"1 arrow = {step_size} iteration(s)", transform=ax.transAxes, color = 'b')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Rosenbrock Function')


    # complete path of the iterates of a point of the grid
    for i in range(0, len(iterates) - 1, step_size):
        start = iterates[i]

        # Assurez-vous que l'indice de fin ne dépasse pas la longueur de la liste
        end_idx = i + step_size
        if end_idx >= len(iterates):
            end_idx = len(iterates) - 1

        end = iterates[end_idx]
        _arrow3D(ax, start[0], start[1], rosenbrock(start), end[0] - start[0], end[1] - start[1], rosenbrock(end) - rosenbrock(start), mutation_scale=20, arrowstyle="-|>", color="b")

    # plot the trajectory of the iterates on the surface of the function in 3D with red arrows
    plt.show()