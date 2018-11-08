import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
import xlsxwriter


def f(x):
    return x**2 / (x + 1)**2


def g(x, y):
    return 4 * y**2 * np.exp(4*x) * (1 - x**3) - 4 * x**3 * y


def d2f(x):
    return 2 / (x + 1)**2 - 8 * x / (x + 1) ** 3 + 6 * x**2 / (x + 1)**4


def trapezoidal_rule(a, b, f, h=0.001):
    res = f(a) + f(b)
    for i in range(1, int((b - a) / h)):
        res += 2 * f(a + h * i)
    return res * h / 2


def find_h_trap(a, b, f, e=0.001):
    h_good = 0.0001
    good_res = trapezoidal_rule(a, b, f, h_good)
    h = 1
    res = trapezoidal_rule(a, b, f, h)
    while abs(res - good_res) > e:
        h = h / 2
        res = trapezoidal_rule(a, b, f, h)
        if abs(res - good_res) < e:
            break
        if res < good_res:
            h += h / 2
        else:
            h -= h / 2
    return h


def trapezoidal_error(a, b, d2f, h):
    m2 = np.max([d2f(x) for x in np.arange(a, b, h)])
    return h**2 * (b - a) / 12 * m2


def simpson(a, b, f, h):
    res = 0
    n = int((b - a) / h)
    for i in range(1, n, 2):
       res += f(a + h * (i - 1)) + 4 * f(a + h * i) + f(a + h * (i + 1))
    res *= h / 3
    return res


def recalculate_h(a, b, h):
    n = int((b - a) / h)
    while n % 4:
        n += 1
    return (b - a) / n


def simpson_error(a, b, d4f, h):
    m4 = np.max([abs(d4f(x)) for x in np.arange(a, b, h)])
    return (b - a) * h**4 / 2880 * m4


def runge_kutta4(a, b, g, h, start_point):
    res = [start_point]
    n = int((b - a - h / 2) / h) + 1
    for k in range(1, n + 1):
        xk = res[k - 1][0]
        yk = res[k - 1][1]
        f1 = g(xk, yk)
        f2 = g(xk + h / 2, yk + h / 2 * f1)
        f3 = g(xk + h / 2, yk + h / 2 * f2)
        f4 = g(xk + h, yk + h * f3)
        res.append((xk + h, yk + h / 6 * (f1 + 2 * f2 + 2 * f3 + f4)))
    return res


def euler(a, b, g, h, start_point):
    res = [start_point]
    n = int((b - a - h / 2) / h) + 1
    for k in range(1, n + 1):
        xk = res[k - 1][0]
        yk = res[k - 1][1]
        res.append((xk + h, yk + h / 2 * (g(xk, yk) + g(xk + h, yk + h * g(xk, yk)))))
    return res


def right(x):
    return 2 * np.exp(1) / (np.exp(x**4) + 2 * np.exp(x**4+4) - 2 * np.exp(4*x+1))


def find_diff_h(a, b, g, start_point, e=0.0001, method='runge'):
    h0 = np.power(e, 1 / 4)
    n = int((b - a) / h0)
    if n % 2:
        n += 1
    h0 = (b - a) / n
    y2 = 1
    y1 = 0
    while 1 / 15 * abs(y2 - y1) >= e:
        if method == 'runge':
            res1 = runge_kutta4(a, a + 2 * h0, g, h0, start_point)
            res2 = runge_kutta4(a, a + 2 * h0, g, 2 * h0, start_point)
        elif method == 'adams':
            res1 = adams(a, a + 2 * h0, g, h0, start_point)
            res2 = adams(a, a + 2 * h0, g, 2 * h0, start_point)

        y1 = res1[-1][1]
        y2 = res2[-1][1]
        h0 /= 2
    return h0


def adams(a, b, g, h, start_point):
    n = int((b - a - h / 2) / h) + 1
    x1 = a + h
    y1 = runge_kutta4(a, x1, g, h, start_point)[-1][1]
    res = [start_point, (x1, y1)]
    for k in range(2, n + 1):
        xk = res[k - 1][0]
        yk = res[k - 1][1]
        xkk = res[k - 2][0]
        ykk = res[k - 2][1]
        res.append((xk + h, yk + h / 2 * (3 * g(xk, yk) - g(xkk, ykk))))
    return res


def save_data(file_name, data):
    workbook = xlsxwriter.Workbook(file_name)
    worksheet = workbook.add_worksheet()
    row = 0
    for col, data in enumerate(data):
        worksheet.write_column(row, col, data)

    workbook.close()


def prepare_data(pts_h, pts_2h):
    n = len(pts_2h)
    data = [np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)]
    data[0] = [t[0] for t in pts_2h]
    data[1] = [pts_h[i][1] for i in range(0, len(pts_h), 2)]
    data[2] = [t[1] for t in pts_2h]
    data[3] = np.abs([data[2][i] - data[1][i] for i in range(n)])
    return data


def result_table(sol, pts_runge, pts_adams):
    n = min(len(pts_runge), len(pts_adams) // 2)
    pts_runge_resized = pts_runge[:n]
    pts_adams_resized = pts_adams[::2]
    pts_adams_resized = pts_adams_resized[:n]
    x = [t[0] for t in pts_runge_resized]
    y_runge = [t[1] for t in pts_runge_resized]
    y_adams = [t[1] for t in pts_adams_resized]
    y_true = [sol(t) for t in x]
    data = [np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)]
    data[0] = x
    data[1] = y_true
    data[2] = y_runge
    data[3] = np.abs([data[2][i] - data[1][i] for i in range(n)])
    data[4] = y_adams
    data[5] = np.abs([data[4][i] - data[1][i] for i in range(n)])
    return data


def main():
    x = sp.Symbol('x')
    y = x**2 / (x + 1)**2
    a = 1
    b = 4
    h = find_h_trap(a, b, f)
    print('Step size for trapezoidal method: ', h)
    h = recalculate_h(a, b, h)
    print('Recalculated step size: ', h)
    res_trapezoidal_h = trapezoidal_rule(a, b, f, h)
    res_trapezoidal_2h = trapezoidal_rule(a, b, f, 2 * h)
    # print(res_trapezoidal, h, trapezoidal_rule(a, b, f, h))
    trap_error_h = trapezoidal_error(a, b, d2f, h)
    trap_error_2h = trapezoidal_error(a, b, d2f, 2*h)
    print('Result of trapezoidal method(h, 2h): ', res_trapezoidal_h, res_trapezoidal_2h)
    print('Error of trapezoidal method with step h: ', trap_error_h)
    print('Error of trapezoidal method with step 2h: ', trap_error_2h)
    res_simpson_h = simpson(a, b, f, h)
    res_simpson_2h = simpson(a, b, f, 2 * h)
    print('Result of simpson method(h, 2h): ', res_simpson_h, res_simpson_2h)
    d4f = sp.lambdify(x, y.diff(x).diff(x).diff(x).diff(x), 'numpy')
    simpson_error_h = simpson_error(a, b, d4f, h)
    simpson_error_2h = simpson_error(a, b, d4f, 2 * h)
    print('Error of simpson method with step h: ', simpson_error_h)
    print('Error of simpson method with step 2h: ', simpson_error_2h)

    #diff calculating
    a = 1
    b = 1.6
    start_point = (1., 2.)
    h = np.float16(find_diff_h(a, b, g, start_point))
    print('Step size for Runge-Cutta method: ', h)
    pts_runge_h = runge_kutta4(a, b, g, h, start_point)
    pts_runge_2h = runge_kutta4(a, b, g, 2 * h, start_point)
    plt.plot([t[0] for t in pts_runge_h], [t[1] for t in pts_runge_h], label='Runge-Cutta')
    plt.plot([t[0] for t in pts_runge_h], [right(t[0]) for t in pts_runge_h], label='Real function')
    h = find_diff_h(a, b, g, start_point, method='adams')

    pts_adams_h = adams(a, b, g, h, start_point)
    pts_adams_2h = adams(a, b, g, 2 * h, start_point)
    plt.plot([t[0] for t in pts_adams_h], [t[1] for t in pts_adams_h], label='Adams')
    #plt.plot([t[0] for t in pts_adams_2h], [t[1] for t in pts_adams_2h])

    pts_euler_h = euler(a, b, g, h, start_point)
    pts_euler_2h = euler(a, b, g, 2 * h, start_point)
    plt.plot([t[0] for t in pts_euler_h], [t[1] for t in pts_euler_h], label='Euler')
    #plt.plot([t[0] for t in pts_euler_2h], [t[1] for t in pts_euler_2h])
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, borderaxespad=0.)
    plt.show()
    data = result_table(right, pts_runge_h, pts_adams_h)
    #save_data('Summary.xlsx', data)


if __name__ == '__main__':
    main()
