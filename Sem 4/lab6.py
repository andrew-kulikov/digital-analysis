import numpy as np
from matplotlib import pyplot as plt


def lagrange(x, y, x0):
    n = len(x)
    res = 0
    for i in range(n):
        b = 1
        a = 1
        for j in range(n):
            if j != i:
                a *= x0 - x[j]
                b *= x[i] - x[j]
        res += (a / b) * y[i]
    return res


def finite_difference(x, y):
    n = len(x)
    M = np.zeros((n, n + 1))
    M[:, 0] = y
    for i in range(1, n):
        for j in range(n - i):
            M[j, i] = M[j + 1, i - 1] - M[j, i - 1]
    return M


def divided_difference(x, y):
    n = len(x)
    M = np.zeros((n, n + 1))
    M[:, 0] = y
    for i in range(1, n):
        for j in range(n - i):
            M[j, i] = (M[j + 1, i - 1] - M[j, i - 1]) / (x[j + i] - x[j])
    return M


def newton(x, y, x0, m=None):
    n = len(x)
    if not m:
        m = divided_difference(x, y)
    res = 0
    coeffs = m[0, :]
    for i in range(n):
        a = 1
        for j in range(i):
            a *= x0 - x[j]
        res += coeffs[i] * a
    return res


def linear_approx(x, y):
    n = len(x)
    a = np.zeros(n)
    b = np.zeros(n)
    for i in range(1, n):
        a[i - 1] = (y[i - 1] - y[i]) / (x[i - 1] - x[i])
        b[i - 1] = y[i] - a[i - 1] * x[i]
    return a, b


def quadratic_approx(x, y):
    n = len(x)
    a = np.zeros((n - 2, 3))
    for i in range(1, n - 1):
        a[i - 1, 2] = (y[i + 1] - y[i - 1]) / ((x[i + 1] - x[i - 1]) * (x[i + 1] - x[i])) - \
                      (y[i] - y[i - 1]) / ((x[i] - x[i - 1]) * (x[i + 1] - x[i]))
        a[i - 1, 1] = (y[i] - y[i - 1]) / (x[i] - x[i - 1]) - a[i - 1, 2] * (x[i] + x[i - 1])
        a[i - 1, 0] = y[i - 1] - a[i - 1, 1] * x[i - 1] - a[i - 1, 2] * x[i - 1] ** 2
    return a


def build_3diag(x):
    n = len(x)
    A = np.zeros(n)
    B = np.zeros(n)
    C = np.zeros(n)
    for i in range(1, n):
        A[i] = x[i] - x[i - 1]
    for i in range(n - 1):
        B[i] = x[i + 1] - x[i]
    for i in range(n):
        C[i] = 2 * (x[i + 1] - x[i])


def cube_splines(x, y):
    n = len(x)
    h = np.zeros(n)
    l = np.zeros(n)
    delta = np.zeros(n)
    lam = np.zeros(n)
    for i in range(n - 1):
        h[i] = x[i + 1] - x[i]
        l[i] = (y[i + 1] - y[i]) / h[i]
    delta[0] = - 1 / 2 * h[1] / (h[0] + h[1])
    lam[0] = 3 / 2 * (l[1] - l[0]) / (h[1] + h[0])
    for i in range(2, n):
        delta[i - 1] = - h[i] / (2 * h[i - 1] + 2 * h[i] + h[i - 1] * delta[i - 2])
    for i in range(2, n):
        lam[i - 1] = (2 * l[i] - 3 * l[i - 1] - h[i - 1] * lam[i - 2]) / (
            2 * h[i - 1] + 2 * h[i] + h[i - 1] * delta[i - 2])

    c = np.zeros(n)
    a = np.zeros(n)
    a[:-1] = y[1:]
    b = np.zeros(n)
    d = np.zeros(n)
    for i in range(n - 1, 0, -1):
        c[i - 1] = delta[i - 1] * c[i] + lam[i - 1]
    for i in range(1, n - 1):
        b[i] = l[i] + 2 / 3 * c[i] * h[i] + 1 / 3 * h[i] * c[i - 1]
        d[i] = (c[i] - c[i - 1]) / (3 * h[i])
    b[0] = l[0] + 2 / 3 * c[0] * h[0]
    d[0] = c[0] / (3 * h[0])
    return a, b, c, d


def get_line_y(x, k, b):
    return k * x + b


def get_parabola_y(x, a, b, c):
    return a * x ** 2 + b * x + c


def get_cube_y(x0, a, b, c, d, xk):
    return a + b * (x0 - xk) + c * (x0 - xk) ** 2 + d * (x0 - xk) ** 3


def main():
    x = np.array([0.235, 0.672, 1.385, 2.051, 2.908])
    y = np.array([1.082, 1.805, 4.280, 5.011, 7.082])
    plt.plot(x, y, 'ro', label='points')
    xpts = np.arange(np.min(x), np.max(x), 0.01)
    ypts_lagrange = np.array([lagrange(x, y, t) for t in xpts])
    plt.plot(xpts, ypts_lagrange, label='Lagrange')
    ypts_newton = np.array([newton(x, y, t) for t in xpts])
    plt.plot(xpts, ypts_newton, label='Newton')
    print("L4(x1+x2) = " + str(lagrange(x, y, x[1] + x[2])))
    print("N4(x1+x2) = " + str(newton(x, y, x[1] + x[2])))
    print(finite_difference(x, y))
    print(divided_difference(x, y))
    a, b = linear_approx(x, y)
    ypts_linear = []
    xpts_linear = []
    for i in range(1, len(x)):
        cur_xpts = np.arange(x[i - 1], x[i], 0.01)
        xpts_linear += list(cur_xpts)
        ypts_linear += [get_line_y(t, a[i - 1], b[i - 1]) for t in cur_xpts]
    plt.plot(xpts_linear, ypts_linear, label='Linear spline')
    a = quadratic_approx(x, y)
    ypts_quadratic = []
    xpts_quadratic = []
    for i in range(1, len(x) - 1, 2):
        cur_xpts = np.arange(x[i - 1], x[i + 1], 0.01)
        xpts_quadratic += list(cur_xpts)
        ypts_quadratic += [get_parabola_y(t, a[i - 1, 2], a[i - 1, 1], a[i - 1, 0]) for t in cur_xpts]
    if i < len(x) - 3:
        cur_xpts = np.arange(x[-2], x[-1], 0.01)
        xpts_quadratic += list(cur_xpts)
        ypts_quadratic += [get_parabola_y(t, a[-1, 2], a[-1, 1], a[-1, 0]) for t in cur_xpts]
    plt.plot(xpts_quadratic, ypts_quadratic, label='Quadratic spline')

    a, b, c, d = cube_splines(x, y)
    xpts_cubic = []
    ypts_cubic = []
    for i in range(len(x) - 1):
        cur_xpts = np.arange(x[i], x[i + 1], 0.01)
        xpts_cubic += list(cur_xpts)
        ypts_cubic += [get_cube_y(t, a[i], b[i], c[i], d[i], x[i + 1]) for t in cur_xpts]

    plt.plot(xpts_cubic, ypts_cubic, label='Cubic spline')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, borderaxespad=0.)
    plt.show()


if __name__ == '__main__':
    main()
