from matplotlib import pyplot as plt
import numpy as np
from sympy import *
from sympy.plotting import plot_parametric
from sympy.utilities.lambdify import lambdastr
from pprint import pprint


def f(x):
    return x * x + np.log(x) - 2


def graph():  
    x = Symbol('x')
    y = Symbol('y')
    p1 = plot_implicit(Eq(sin(x+y)-1.5*x, 0.1), (x, -2, 2), (y, -2, 2))
    p2 = plot_implicit(Eq(x**2+y**2, 1), (x, -2, 2), (y, -2, 2))
    p1.extend(p2)
    p1.show()
    

def newton_eq(f, fstr, x0, e=0.00001, max_iterations=10000):
    print('Newton method is running')
    x = Symbol('x')
    df = lambdify(x, fstr.diff(x), 'numpy')
    xk = x0 - f(x0) / df(x0) 
    for k in range(max_iterations):
        xp = xk
        xk -= f(xk) / df(xk)
        print('Iteration #' + str(k + 1) + ' Current solution: ' + str(xk))
        if abs(xp - xk) < e:
            break
    return xk, k


def chords(f, a, b, e=0.001, max_iterations=10000):
    print('Chords method is running')
    xn = b
    xp = a
    i = 0
    while abs(f(xn)) > e:
        tmp = xn
        xn = xp - f(xp) / (f(xn) - f(xp)) * (xn - xp)
        xp = tmp
        i += 1
        print('Iteration #' + str(i) + ' Current solution: ' + str(xn))
        if i > max_iterations:
            break
    return xn, i


def iterations(x0, F, e=0.001, max_iterations=10000):
    print('Iterations method is running')
    xp = np.copy(x0)
    for i in range(max_iterations):
        xk = np.copy(xp)
        for j in range(len(xk)):
            xk[j] = F[j](*xp)
        print('Iteration #' + str(i + 1) + ' Current solution: ' + str(xk))
        if np.max(np.abs(xk - xp)) < e:
            break
        xp = xk
    return xp, i + 1


def build_F(exprs):
    x, y = symbols('x y')
    F = []
    for i in range(len(exprs)):
        F.append(lambdify((x, y), exprs[i], 'numpy'))
    return F


def build_jacobian(syms, funcs):
    J = []
    for i in range(len(funcs)):
        J.append([])
        for sym in syms:
            J[i].append(lambdify(syms, funcs[i].diff(sym), 'numpy'))
    return J


def eval_jacobian(J, vals):
    rows, cols = J.shape
    M = np.zeros(J.shape)
    for i in range(rows):
        for j in range(cols):
            M[i, j] = J[i, j](*vals)
    return M


def eval_F(F, vals):
    F1 = np.zeros(F.shape)
    for i in range(len(F)):
        F1[i] = F[i](*vals)
    return F1


def newton_syst(J, F, x0, e=0.001, max_iterations=10000):
    print('Newton method is running')
    xp = np.copy(x0)
    xk = np.copy(x0)
    for i in range(max_iterations):
        xp = np.copy(xk)
        xk = xk - np.dot(
                np.linalg.inv(eval_jacobian(J, xk)),
                eval_F(F, xk))
        print('Iteration #' + str(i + 1) + ' Current solution: ' + str(xk))
        if np.max(np.abs(xp - xk)) < e:
            break
    return xk, i + 1


def newton_syst_mod(J, F, x0, e=0.001, max_iterations=10000):
    print('Modified Newton method is running')
    xp = np.copy(x0)
    xk = np.copy(x0)
    J0 = np.linalg.inv(eval_jacobian(J, xk))

    for i in range(max_iterations):
        xp = np.copy(xk)
        xk = xk - np.dot(J0, eval_F(F, xk))
        print('Iteration #' + str(i + 1) + ' Current solution: ' + str(xk))
        if np.max(np.abs(xp - xk)) < e:
            break
    return xk, i + 1


def main():
    a = 1
    b = 1.5
    ans, iters_chords = chords(f, a, b)
    x, y = symbols('x y')
    fstr = x**2 + log(x) - 2
    plot(x**2 + log(x) - 2, (x, 0.01, 4))
    ans1, iters_newton_eq = newton_eq(f, fstr, 1.25)
    print(ans, ans1)
    print(iters_chords, iters_newton_eq)
    graph()
    F = build_F([2/3*(sin(x+y) - 0.1), x**2+y**2-1+y])
    x0 = np.zeros(2)
    x0, iters_iter = iterations([-0.5, -0.4], F)
    print('Iterations method for system: ')
    print('Amount of iterations: ' + str(iters_iter))
    print('Answer: ' + str(x0))
    J = np.array(build_jacobian([x, y], [sin(x+y)-0.1-1.5*x, x**2+y**2-1]))
    F = np.array(build_F([sin(x+y)-0.1-1.5*x, x**2+y**2-1]))
    x0, iters_newton_sys = newton_syst(J, F, [0.5, 0.75])
    print('Newton method for system: ')
    print('Amount of iterations: ' + str(iters_newton_sys))
    print('Answer: ' + str(x0))
    x0, iters_newton_sys_mod = newton_syst_mod(J, F, [0.5, 0.75])
    print('Modified newton method for system: ')
    print('Amount of iterations: ' + str(iters_newton_sys_mod))
    print(x0)
    
    

if __name__ == '__main__':
    main()
    