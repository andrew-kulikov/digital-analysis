import numpy as np
import sympy as sp


def collocation(diff_eq, x1, x2):
    x, a1, a2 = sp.symbols('x a1 a2')
    eq1 = diff_eq.subs({x: x1})
    eq2 = diff_eq.subs({x: x2})

    print('Given system: ')
    print(eq1)
    print(eq2, '\n')
    return sp.solve([eq1, eq2], [a1, a2])


def integral_lsm(diff_eq, x_inf=-1, x_sup=1):
    x, a1, a2 = sp.symbols('x a1 a2')
    integral = sp.integrate(diff_eq**2, (x, x_inf, x_sup))
    print('I(a1, a2): ')
    print(integral)

    eq1 = sp.diff(integral, a1)
    eq2 = sp.diff(integral, a2)
    print('Given system: ')
    print(eq1)
    print(eq2, '\n')
    return sp.solve([eq1, eq2], [a1, a2])


def discrete_lsm(diff_eq, n=100, x_start=-1, x_end=1):
    x, a1, a2 = sp.symbols('x a1 a2')

    n -= 1
    h = (x_end - x_start) / n
    S = diff_eq.subs({x: x_start})**2
    for x_step in np.arange(x_start + h, x_end + h / 2, h):
        S += diff_eq.subs({x: x_step})**2

    print('Sum:', S)
    eq1 = sp.diff(S, a1)
    eq2 = sp.diff(S, a2)
    print('Given system: ')
    print(eq1)
    print(eq2, '\n')
    return sp.solve([eq1, eq2], [a1, a2])


def galerkin(diff_eq, phi1, phi2, x_inf=-1, x_sup=1):
    x, a1, a2 = sp.symbols('x a1 a2')
    eq1 = sp.integrate(diff_eq * phi1, (x, x_inf, x_sup))
    eq2 = sp.integrate(diff_eq * phi2, (x, x_inf, x_sup))
    print('Given system: ')
    print(eq1)
    print(eq2, '\n')
    return sp.solve([eq1, eq2], [a1, a2])


def main():
    k = int(input("Enter variant number: "))
    a_inp = np.sin(k)
    b_inp = np.cos(k)
    x, y, a, b, a1, a2, phi0, phi1, phi2 = sp.symbols('x y a b a1 a2 phi0 phi1 phi2')

    phi0 = 0
    phi1 = (1 - x**2)
    phi2 = x**2 * (1 - x**2)

    y2 = phi0 + a1 * phi1 + a2 * phi2
    residual = a * sp.diff(sp.diff(y2, x), x) + (1 + b * x**2) * y2 + 1
    residual = residual.subs({a: a_inp, b: b_inp})

    print('-' * 20)
    print('y2: {y}\nresidual: {residual}'.format(y=y2, residual=residual))
    print('-' * 20)

    print('Collocation method result:', collocation(residual, -1/2, 0), '\n')
    print('-' * 20)

    print('Integral lsm:', integral_lsm(residual, -1, 1))
    print('-' * 20)

    print('Discrete lsm, n = 3:', discrete_lsm(residual, 3, -1, 1))
    print('-' * 20)

    print('Galerkin:', galerkin(residual, phi1, phi2))
    print('-' * 20)


if __name__ == '__main__':
    main()
