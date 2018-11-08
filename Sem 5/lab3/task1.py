import numpy as np
import sympy as sp
from matplotlib import pyplot as plt


def get_params(k, ua, ub, variant):
    params = [(1, k, ua, ub),
              (2, 2 * k, ua, ub),
              (0.1, 0.1 * k, ua, ub),
              (1, 1 / k, ua, ub),
              (1, k, -ua, ub),
              (1, k, ua, -ub),
              (1, k, -ua, -ub)]
    return params[variant - 1]


def plot_solution(foo, x1, x2, label):
    x = sp.symbols('x')
    args = np.linspace(x1, x2, 200)
    y = [foo.subs({x: t}) for t in args]
    plt.plot(args, y, label=label)


def solve(k, a, b, ua, ub, f):
    x, c1, c2 = sp.symbols('x c1 c2')
    u = -sp.integrate(sp.integrate(f, x) / k, x) + c1 * x + c2
    sol = sp.solve([u.subs({x: a}) - ua, u.subs({x: b}) - ub], (c1, c2))
    return u.subs({c1: sol[c1], c2: sol[c2]})


def main():
    x, k, f = sp.symbols('x k f')
    f = sp.exp(2 * x)
    k = sp.exp(x)

    a = 0.5
    b = 1.5
    ua = 1
    ub = 5

    for var in range(1, 4):
        c, cur_k, cur_ua, cur_ub = get_params(k, ua, ub, var)
        u = solve(cur_k, a, b, cur_ua, cur_ub, f)
        plot_solution(u, a, b, 'k={}'.format(var))

    plt.legend()
    plt.show()

    for var in [1, 4]:
        c, cur_k, cur_ua, cur_ub = get_params(k, ua, ub, var)
        u = solve(cur_k, a, b, cur_ua, cur_ub, f)
        plot_solution(u, a, b, 'k={}'.format(var))

    plt.legend()
    plt.show()

    for var in [5, 6, 7]:
        c, cur_k, cur_ua, cur_ub = get_params(k, ua, ub, var)
        u = solve(cur_k, a, b, cur_ua, cur_ub, f)
        plot_solution(u, a, b, 'k={}'.format(var))

    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
