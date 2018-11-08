import sympy as smp
import numpy as np
from matplotlib import pyplot as plt


x = smp.symbols('x')
h = 0.01
tau = 0.05
start = 0
finish = 1


def solve(t):
    Y = []
    A = []
    B = []
    k_x = x
    f_x = 1 / x
    i = 0
    for xi in np.arange(start + h, finish, h):
        if t != 0:
            a = (-2 * tau * k_x).subs(x, xi)
            b = (h**2 + 4 * tau * k_x - tau * h * k_x.diff(x)).subs(x, xi)
            c = (-h**2 + smp.diff(k_x, x) * tau * h - 2 * tau * k_x).subs(x, xi)
            d = (f_x * (1 - smp.exp(-t)) * 2 * tau * h**2).subs(x, xi)

            if i == 0:
                Ai = - c / b
                Bi = d / b
            else:
                Ai = -c / (b + a * A[-1])
                Bi = (d - a * B[-1]) / (b + a * A[-1])

            A.append(Ai)
            B.append(Bi)
        else:
            Y.append(xi ** 2)
        i += 1

    if t == 0:
        Y.append(finish ** 2)
        return Y

    Y = [0] * (i + 1)

    while i > 1:
        i -= 1
        Y[i] = A[i] * Y[i + 1] + B[i]

    return Y


def main():
    t = [0.5 * tau, 20 * tau, 200 * tau]
    x_arr = np.arange(start, finish, h)

    for t_ in np.arange(0, 101 * tau, 4 * tau):
        plt.plot(x_arr, solve(t_), label="t = {}".format(t_))
        plt.show()


if __name__ == "__main__":
    main()
