import numpy as np
import matplotlib.pyplot as plt

from utils import rosenbrock, rosenbrock_prime, _arrow3D
from line_search import golden_section_search, wolfe_search

def gradient_descent_variable_step(x_init, a, b, f, f_prime, line_search, tolerance=1e-5, max_iteration=10000):
    """
    Use the gradient descent method with variable step to find the minimum of a function f.
    """

    x = x_init
    iterates = [x_init]
    residual = 2 * tolerance
    learning_rate = 1e-3
    line_search_tolerance = 1e-4

    while np.linalg.norm(residual) > tolerance and len(iterates) < max_iteration:
        f_alpha = lambda alpha: f(x - alpha * np.array(f_prime(x)))
        f_alpha_prime = lambda alpha: f_prime(x + alpha * np.array(f_prime(x))) * np.array(f_prime(x))

        descent_direction = np.array(f_prime(x))
        if line_search.__name__ == 'golden_section_search':
            learning_rate = golden_section_search(f_alpha, a, b)
        elif line_search.__name__ == 'wolfe_search':
            if isinstance(x, float):
                line_search_tolerance = 1e-5
            learning_rate = wolfe_search(f_alpha, f_alpha_prime, x, -descent_direction, tolerance=line_search_tolerance)
        delta = - learning_rate * descent_direction
        
        x = x + delta
        residual = np.linalg.norm(x - iterates[-1])
        iterates.append(x)

    return iterates, x


if __name__ == '__main__':
    step_size = 100
    f = lambda x: np.exp(x) - 2 * x
    f_prime = lambda x: np.exp(x) - 2
    iterates_golden_section, x_min_golden_section = gradient_descent_variable_step(-1.5, -2, 2, f, f_prime, golden_section_search)
    iterates_wolfe, x_min_wolfe = gradient_descent_variable_step(-1.5, 0, 2, f, f_prime, wolfe_search)
    print("For the function f(x) = exp(x) - 2x, in 1D: ")
    print(f"Minimum of f using the golden section method: {x_min_golden_section}")
    print(f"No. of iterations using the golden section method: {len(iterates_golden_section)}")
    print(f"Minimum of f using the Wolfe method: {x_min_wolfe}")
    print(f"No. of iterations using the Wolfe method: {len(iterates_wolfe)}")


    # plot the function f in 2D
    x_vals = np.linspace(-2, 2, 100)
    y_vals = f(x_vals)

   
    for i in range(len(iterates_wolfe) - 1):
        start = iterates_wolfe[i]
        end = iterates_wolfe[i + 1]
        plt.arrow(start, f(start), end - start, f(end) - f(start), color='blue',shape='full', linestyle='--', lw=1, length_includes_head=True, head_width=0.05)

    for i in range(len(iterates_golden_section) - 1):
        start = iterates_golden_section[i]
        end = iterates_golden_section[i + 1]
        plt.arrow(start, f(start), end - start, f(end) - f(start), color='green', shape='full', linestyle='--', lw=1, length_includes_head=True, head_width=0.05)
    plt.plot(x_vals, y_vals, label='Function f(x) = exp(x) - 2x')
    plt.plot([], [], color='blue', linestyle="--", label='Gradient descent with Wolfe')
    plt.plot([], [], color='green', linestyle="--", label='Gradient descent with golden section')

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Optimization Visualization for the Gradient algorithm with golden section')
    plt.legend()

    print("-------------------------------------")
    print("For the Rosenbrock function, in 2D: ")
    iterates_wolfe_2D, x_wolfe_2D = gradient_descent_variable_step(np.array([1.8, -1.7]), 0, 2, rosenbrock, rosenbrock_prime, wolfe_search)
    print(f"Mimimum of the Rosenbrock function using the Wolfe method: {x_wolfe_2D}")
    print(f"No. of iterations using the Wolfe method: {len(iterates_wolfe_2D)}")


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


    for i in range(0, len(iterates_wolfe_2D) - 1, step_size):
        start = iterates_wolfe_2D[i]

        end_idx = i + step_size
        if end_idx >= len(iterates_wolfe_2D):
            end_idx = len(iterates_wolfe_2D) - 1

        end = iterates_wolfe_2D[end_idx]
        _arrow3D(ax, start[0], start[1], rosenbrock(start), end[0] - start[0], end[1] - start[1], rosenbrock(end) - rosenbrock(start), mutation_scale=20, arrowstyle="-|>", color="b")

    plt.show()