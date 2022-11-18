import numpy as np

def supnorm(f, step, a, b, deg):
    """
    

    Args:
        f (_type_): _description_
        NumNodes (_type_): _description_
        a (_type_): _description_
        b (_type_): _description_
        deg: 

    Output:
        max(fx) : supremum norm
    """
    x = np.linspace(a, b, step)
    fx = np.abs(f(x, deg))
    return max(fx)

def LagrangeInterpolation(fx, x, z):
    """

    Args:
        fx (_type_): _description_
        x (_type_): _description_
        z (_type_): _description_

    Output:
        v : _description_
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

# def NewtonLagrangeInterpolation(fx, x, z):
#     n = len(x)
#     m = len(z)
#     d = np.zeros((n, n))
#     v = np.empty(m)
#     for k in range(0, n): d[k][0] = fx[k]
#     for k, l in zip(range(0, n), range(1, n)): d[k][l] = 0
#     for k in range(1, n):
#         for l in range(k, n):
#             d[l][k] = (d[l][k-1] - d[l-1][k-1]) / (x[l]-x[l-k])
#     for j in range(0, m):
#         v[j] = d[n][n]
#         for k in range(1, n):
#             v[j] = v[j] * (z[j] - x[n-k]) + d[n-k][n-k]
#     return v

# def HermiteInterpolation(fx, dfx, x, z):
#     n = len(x)
#     m = len(z)
#     v = np.empty(m)
#     for j in range(0, m):
#         for k in range(0, n):
#             delta = 1
#             nu = 1
#             l1 = 0
#             for i in range(0, n):
#                 if (i!=k):
#                     nu = nu * (z[j] - x[i])
#                     delta = delta * (x[k] - x[i])
#                     l1 = l1 + 1 / (x[k] - x[i])
#             l2 = (nu/delta)**2
#             eta = (z[j] - x[k]) * l2
#             h = (1 - 2 * l1 * (z[j] - x[k])) * l2
#             v[j] = v[j] + fx[k] * h + dfx[k] * eta
#     return v