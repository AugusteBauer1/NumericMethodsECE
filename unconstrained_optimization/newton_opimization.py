import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from linear_systems.direct_methods_linear_systems import LU_solve
from utils import Arrow3D, _arrow3D, rosenbrock, rosenbrock_prime, hessian_rosenbrock


def gradient_newton(max_iteration,x_init):
    """
    Use the Newton method to find the minimum of a function f in R2.
    Complexity: O(n^3), where n is the number of iterations until convergence.
    """

    # first iteration
    x = x_init
    
    # next iteration until a certain number of iterations
    iterates = [x_init, x]
    while len(iterates) < max_iteration:
        delta = LU_solve(np.array(hessian_rosenbrock(x)), -np.array(rosenbrock_prime(x)))
        x = x + delta
        iterates.append(x)

    return iterates, x

if __name__ == '__main__':
    step_size = 1

    # test the gradient_newton function with the Rosenbrock function for a maximum of 100 iterations and an initial point (-1.2, 1.5)
    iterates, x = gradient_newton(100, [-1.2, 1.5])
    print(f"Root: {x}")

    # first print the function around he point (1, 1) in 3D
    x1 = np.linspace(-2, 2, 50)
    x2 = np.linspace(-2, 2, 50)
    x1, x2 = np.meshgrid(x1, x2)
    grid = np.array([x1, x2])
    z = rosenbrock(grid)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x1, x2, z, cmap='viridis', edgecolor='none')
    ax.text2D(-0.27, 0, f"1 arrow = {step_size} iteration(s)", transform=ax.transAxes, color = 'red')
    plt.xlabel('x1')
    plt.ylabel('x2')    
    plt.title('Rosenbrock Function')


    # complete path of the iterates of a point of the grid
    
    for i in range(len(iterates) - 1):
        start = iterates[i]
        end_idx = min(i + step_size, len(iterates) - 1)
        end = iterates[end_idx]
        _arrow3D(ax, start[0], start[1], rosenbrock(start), end[0] - start[0], end[1] - start[1], rosenbrock(end) - rosenbrock(start), mutation_scale=20, arrowstyle="-|>", color="r")

    # plot the trajectory of the iterates on the surface of the function in 3D with red arrows
    plt.show()
            
    


    



