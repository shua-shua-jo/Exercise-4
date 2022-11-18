import numpy as np

def supnorm(f, step, a, b, deg):
    """
    Supremum norm of f

    Args:
        f : vector
            polynomial function
        step : integer
            number of nodes
        a : real number
            lower bound
        b : real number
            upper bound
        deg : integer
            degree of polynomial

    Output:
        max(fx) : supremum norm
    """
    x = np.linspace(a, b, step)
    fx = np.abs(f(x, deg))
    return max(fx)

def LagrangeInterpolation(fx, x, z):
    """
    Lagrange Interpolation Polynomial

    Args:
        fx : vector
            function value at x
        x : vector
            interpolation nodes
        z : vector
            points to be evaluated

    Output:
        v : vector
            values of the lagrange interpolation polynomial
    """
    m = len(z)
    n = len(x)
    v = np.empty(m)
    for j in range(0, m):
        v[j] = 0
        for k in range(0, n):
            l = 1
            for i in range(0, n):
                if i != k:
                    l = l*(z[j]-x[i])/(x[k]-x[i])
            v[j] = v[j] + fx[k]*l
    return v