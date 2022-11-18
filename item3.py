import math
import numpy as np
import polyinterp as poly_eval
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-pastel')

def f(x):
    return 1 / (1 - x**2 + x**4)

def Pn(z, a, b, deg):
    x = chebyshev(a, b, deg)
    fx = f(x)
    Pz = poly_eval.LagrangeInterpolation(fx, x, z)
    return Pz

def chebyshev(a, b, n):
    x = np.empty(n)
    t1 = (a + b) / 2
    t2 = (b - a) / 2
    for k in range(0, n):
        x[k] = t1 + t2 * math.cos((2 * k + 1) * math.pi / (2 * n + 2))
    return x

def g1(x, deg):
    return f(x) - Pn(x, -5, ub1, deg)
def g2(x, deg):
    return f(x) - Pn(x, ub1, ub2, deg)
def g3(x, deg):
    return f(x) - Pn(x, ub2, 5, deg)

# subintervals
ub1 = -5+(10/2.68)
ub2 = 5-(10/2.68)

# [-5,-1.667]
z1 = np.linspace(-5, ub1, 100) 
fz1 = f(z1)

# [-1.667,1.667]
z2 = np.linspace(ub1, ub2, 100) 
fz2 = f(z2)

# [1.667,5]
z3 = np.linspace(ub2, 5, 100) 
fz3 = f(z3)

# n = 10
degree = 11

sup1 = poly_eval.supnorm(g1, 1000, -5, ub1, degree)
sup2 = poly_eval.supnorm(g2, 1000, ub1, ub2, degree)
sup3 = poly_eval.supnorm(g3, 1000, ub2, 5, degree)
supnorm = max(sup1, sup2, sup3)

print(f"supnorm at [-5.00, {round(ub1,2)}]\t: \t", sup1)
Pz1 = Pn(z1, -5, ub1, degree)

print(f"supnorm at [{round(ub1,2)}, {round(ub2,2)}]\t: \t", sup2)
Pz2 = Pn(z2, ub1, ub2, degree)

print(f"supnorm at [{round(ub2,2)}, 5.00]\t\t: \t", sup3)
Pz3 = Pn(z3, ub2, 5, degree)

plt.rcParams['figure.figsize'] = [13, 7]

print(f"supnorm = {supnorm} for 3 subintervals")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)

fig.suptitle('Function f and Lagrange Polynomial P_z in different subintervals')
ax1.plot(z1, fz1, 'k', z1, Pz1, 'b')
ax1.set_title('[-5, -1.667]')
ax2.plot(z2, fz2, 'k', z2, Pz2, 'b')
ax2.set_title('[-1.667, 1.667]')
ax3.plot(z3, fz3, 'k', z3, Pz3, 'b')
ax3.set_title('[1.667, 5]')

fig.legend(['f(x)', 'P_z(x)'])

for ax in fig.get_axes():
    ax.grid()
    
plt.show()