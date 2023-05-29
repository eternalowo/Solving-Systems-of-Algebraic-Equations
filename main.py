import numpy as np


def f1(x, y):
    """First function of first system"""
    return 0.8 - np.cos(y + 0.5)


def f2(x, y):
    """Second function of first system"""
    return (np.sin(x) - 1.5) / 2


def fixed_point_iteration(x0, y0, eps=1e-6, max_iter=1000):
    """Algorithm of fixed-point iteration method"""
    x, y = x0, y0
    iter_nums = 0
    for i in range(max_iter):
        iter_nums += 1
        x_new, y_new = f1(x, y), f2(x, y)

        if np.abs(x_new - x) < eps and np.abs(y_new - y) < eps:
            break

        x, y = x_new, y_new
    else:
        print("Not enough iterations.")
        return None

    return x, y, iter_nums


x0, y0 = 1, 1

solution = fixed_point_iteration(x0, y0)
if solution is not None:
    print(
        f"Solution of first system of equations by fixed-point iteration method, x = {solution[0]:.6f},"
        f" y = {solution[1]:.6f}, number of iterations: {solution[2]}")


def equations(vars):
    """Second system of equations"""
    x, y = vars
    eq1 = np.sin(x + y) - 1.4 * x
    eq2 = x ** 2 + y ** 2 - 1
    return np.array([eq1, eq2])


def jacobian(vars):
    """Calculate Jacobian for second system"""
    x, y = vars
    j11 = np.cos(x + y) - 1.4
    j12 = np.cos(x + y)
    j21 = 2 * x
    j22 = 2 * y
    return np.array([[j11, j12], [j21, j22]])


# Initial approximation
x0 = np.array([0.5, 0.5])

eps = 1e-15

max_iter = 100

# Simultaneous quadratic approximation (SIMQ)
iter_nums = 0
for i in range(max_iter):
    iter_nums += 1
    f = equations(x0)
    J = jacobian(x0)
    dx = np.linalg.solve(J, -f)
    solution = x0 + dx
    if np.linalg.norm(dx) < eps:
        print(f"Solution of second system of equations by Newton method using SIMQ, x = {solution[0]:.6f},"
              f" y = {solution[1]:.6f}, number of iterations: {iter_nums}")
        break
    else:
        x0 = solution

iter_nums = 0
for i in range(max_iter):
    iter_nums += 1
    f = equations(x0)
    J = jacobian(x0)
    J_inv = np.linalg.inv(J)
    dx = -J_inv.dot(f)
    solution = x0 + dx
    if np.linalg.norm(dx) < eps:
        print(f"Solution of second system of equations by Newton method using MINV, x = {solution[0]:.6f},"
              f" y = {solution[1]:.6f}, number of iterations: {iter_nums}")
        break
    else:
        x0 = solution
