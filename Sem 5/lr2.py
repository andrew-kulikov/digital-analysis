import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import math

# x
value = Symbol('value')
# y
funct = Symbol('funct')
# dy
dfunct = Symbol('dfunct')


def method_progonki1(h_):
    x0 = -1
    xn = 1
    x, y, A, B = [], [], [], []
    h = h_
    k = 6

    for v in np.arange(x0, xn + h, h):
        x.append(v)
        y.append(0)
        A.append(0)
        B.append(0)

    a = math.sin(k)
    b = math.cos(k)

    ai = a
    b1 = h**2 * (1 + b * x0**2) - 2 * a
    ci = a
    di = (-1)*(h**2)
    A[0] = (-1)*ci / b1
    B[0] = di / b1

    for i in range(1, len(x) - 1):
        bi = h**2 * (1 + b * x[i]**2) - 2 * a
        A[i] = -ci / (bi + ai * A[i - 1])
        B[i] = (di - ai * B[i - 1]) / (bi + ai * A[i - 1])

    bi = h**2 * (1 + b * (xn - h)**2) - 2 * a
    A[-1] = 0
    B[-1] = (di - ai * B[-2]) / (bi + ai * A[-2])

    for i in reversed(range(1, len(x) - 2)):
        y[i] = A[i] * y[i + 1] + B[i]

    return x, y


def method_progonki2(h):
    x0 = 0
    xn = 1
    x = []
    y = []
    A = []
    B = []
    for i, v in enumerate(np.arange(x0, xn + h, h)):
        x.append(v)
        y.append(0)
        A.append(0)
        B.append(0)

    b1 = -2 / h**2 + 5 * (1 + cos(x[0])**2)
    c1 = 1 / h**2 + cos(x[0]) / (2 * h)
    d1 = 10 / (1 + 0.5 * x[0]**2)
    A[0] = (-1) * c1 / b1
    B[0] = d1 / b1

    for i in range(1, len(x) - 1):
        ai = 1 / h**2 - cos(x[i]) / (2 * h)
        bi = -2 / h ** 2 + 5 * (1 + cos(x[i]) ** 2)
        ci = 1 / h ** 2 + cos(x[i]) / (2 * h)
        di = 10 / (1 + 0.5 * x[i] ** 2)
        A[i] = -ci / (bi + ai * A[i - 1])
        B[i] = (di - ai * B[i - 1]) / (bi + ai * A[i - 1])

    ai = 1 / h ** 2 - cos(x[-1]) / (2 * h)
    bi = -2 / h ** 2 + 5 * (1 + cos(x[-1]) ** 2)
    di = 10 / (1 + 0.5 * x[-1] ** 2)
    A[-1] = 0
    B[-1] = (di - ai * B[-2]) / (bi + ai * A[-2])

    for i in reversed(range(1, len(x) - 2)):
        y[i] = A[i] * y[i + 1] + B[i]

    return x, y


def solution_task3(f, a, b, h_, f_0, f_n):
    x = []
    y = []
    h = h_
    for i, v in enumerate(np.arange(a, b + h, h)):
        x.append(v)
        y.append(Symbol('y{}'.format(i)))
    s = []
    for i in range(1, len(x) - 1):
        s.append((y[i + 1] - 2 * y[i] + y[i - 1]) / (h ** 2) -
                 f.subs(value, x[i]).subs(funct, y[i]).subs(dfunct, (y[i + 1] - y[i - 1]) / (2 * h)))
    s.append(f_0.subs(funct, y[0]).subs(dfunct, (-y[2] + 4 * y[1] - 3 * y[0]) / (2 * h)))
    s.append(f_n.subs(funct, y[-1]).subs(dfunct, (3 * y[-1] - 4 * y[-2] + y[-3]) / (2 * h)))
    sol = next(iter(linsolve(s, y)))
    print(*s, sep='\n')
    return x, sol


def method_progonki4(h):
    a = 0
    b = 2.2
    c = 1.125
    k1 = 0.5
    k2 = 1.8
    q1 = 3.5
    q2 = 7.8
    x0 = a
    xn = b
    x = []
    y = []
    A = []
    B = []
    k = []
    q = []
    for i, v in enumerate(np.arange(x0, xn + h, h)):
        x.append(v)
        y.append(0)
        A.append(0)
        B.append(0)
        if v < c:
            k.append(k1)
            q.append(q1)
        else:
            k.append(k2)
            q.append(q2)

    b1 = q1 * h**2 + 2 * k1
    c1 = (-1) * k1
    d1 = 10 * x0**2 * (2.5 - x0) * h**2
    A[0] = (-1) * c1 / b1
    B[0] = d1 / b1

    for i in range(1, len(x) - 1):
        ai = (-1) * k[i]
        bi = q[i] * h**2 + 2 * k[i]
        ci = (-1) * k[i]
        di = 10 * x[i]**2 * (2.5 - x[i]) * h**2
        A[i] = -ci / (bi + ai * A[i - 1])
        B[i] = (di - ai * B[i - 1]) / (bi + ai * A[i - 1])

    ai = (-1) * k2
    bi = q2 * h**2 + 2 * k2
    di = 10 * xn**2 * (2.5 - xn) * h**2
    A[-1] = 0
    B[-1] = (di - ai * B[-2]) / (bi + ai * A[-2])

    for i in reversed(range(1, len(x) - 2)):
        y[i] = A[i] * y[i + 1] + B[i]

    return x, y


def write_file(filename, x, y1, y2):
    file = open(filename, 'w')
    j = 0
    for i in range(1, len(x) - 1):
        if i % 2 == 0:
            j += 1
        file.write(str(x[i]) + "\t" + str(y1[j]) + "\t" + str(y2[i]) + "\n")
    file.close()


def task1(filename):
    print("Precession and step for task1:\n")
    precision = 1
    h = 0.25
    e = 0.001
    x1, y1 = method_progonki1(2 * h)
    x2, y2 = method_progonki1(h)

    while precision > e:
        precision = max([abs(y2[2 * i] - y1[i]) for i in range(len(y1))])
        x1, y1 = x2, y2
        h /= 2
        x2, y2 = method_progonki1(h)
        print('precision', precision)
        print('h', h)

    x, y = method_progonki1(h)
    write_file(filename, x2, y1, y2)
    plt.plot(x, y)
    plt.show()
    print("\n")


def task2(filename):
    print("Precession and step for task2:\n")
    precision = 1
    h = 0.25
    e = 0.05
    x1, y1 = method_progonki2(2 * h)
    x2, y2 = method_progonki2(h)

    while precision > e:
        precision = max([abs(y2[2 * i] - y1[i]) for i in range(len(y1))])
        x1, y1 = x2, y2
        h /= 2
        x2, y2 = method_progonki2(h)
        print('precision', precision)
        print('h', h)

    x, y = method_progonki2(h / 2)
    write_file(filename, x2, y1, y2)
    plt.plot(x, y)
    plt.show()
    print("\n")


def task3(filename):
    print("Precession and step for task3:\n")
    f = value * funct - 2 * value
    precision = 1
    h = 0.25
    e = 0.1
    #x1, y1 = solution_task3(f, 1.5, 3.5, h * 2, funct - 2 * dfunct - 4.5, dfunct - 3)
    x2, y2 = solution_task3(f, 1.5, 3.5, h, funct - 2 * dfunct - 4.5, dfunct - 3)

    #while precision > e:
    #    precision = max([abs(y2[2 * i] - y1[i]) for i in range(len(y1))])
    #    x1, y1 = x2, y2
    #    h /= 2
    #    x2, y2 = solution_task3(f, 1.5, 3.5, h, funct - 2 * dfunct - 4.5, dfunct - 3)
    #    print('precision', precision)
    #    print('h', h)

    #x, y = solution_task3(f, 1.5, 3.5, h / 2, funct - 2 * dfunct - 4.5, dfunct - 3)
    #write_file(filename, x2, y1, y2)
    plt.plot(x, y)
    plt.show()
    print("\n")


def task4(filename):
    print("Precession and step for task4:\n")
    precision = 1
    h = 0.022
    e = 0.001
    x1, y1 = method_progonki4(2 * h)
    x2, y2 = method_progonki4(h)

    while precision > e:
        precision = max([abs(y2[2 * i] - y1[i]) for i in range(len(y1) - 1)])
        x1, y1 = x2, y2
        h /= 2
        x2, y2 = method_progonki4(h)
        print('precision', precision)
        print('h', h)

    x, y = method_progonki4(h)
    write_file(filename, x2, y1, y2)
    plt.plot(x, y)
    plt.show()
    print("\n")


def main():
    # task1("./task1.txt")
    # task2("./task2.txt")
    task3("./task3.txt")
    # task4("./task4.txt")


if __name__ == "__main__":
    main()
