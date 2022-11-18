import numpy as np
import polyinterp as poly_eval
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-pastel')

def f(x):
    """
    Polynomial Function

    Args:
        x : vector

    Returns:
        f : real number
            function value at x
    """
    return 1 / (1 - x**2 + x**4)

def Pn(z, deg):
    """
    Polynomial interpolation of f using Lagrange

    Args:
        z : vector
        deg : integer
            degree of polynomial

    Returns:
        Pz: vector
            lagrange interpolation polynomial
    """
    x = np.linspace(-5, 5, deg)
    fx = f(x)
    Pz = poly_eval.LagrangeInterpolation(fx, x, z)
    return Pz

def g(x, deg):
    """
    Function of supnorm
    
    Args:
        x : vector
        deg : integer
            degree of polynomial

    Returns:
        g: vector
            difference of f - Pn
    """
    return f(x) - Pn(x, deg)


z = np.linspace(-5, 5, 100) 
fz = f(z)

# n = 5
degree = 6
print("supnorm (n=5): ", poly_eval.supnorm(g, 1000, -5, 5, degree))
Pz1 = Pn(z, degree)

# n = 10
degree = 11
print("supnorm (n=10): ", poly_eval.supnorm(g, 1000, -5, 5, degree))
Pz2 = Pn(z, degree)

# n = 20
degree = 21
print("supnorm (n=20): ", poly_eval.supnorm(g, 1000, -5, 5, degree))
Pz3 = Pn(z, degree)

# plotting the function and polynomial interpolation

plt.rcParams['figure.figsize'] = [13, 7]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)

fig.suptitle('Function f and Lagrange Polynomial P_z with different degrees n')
ax1.plot(z, fz, 'k', z, Pz1, 'b')
ax1.set_title('n = 5')
ax2.plot(z, fz, 'k', z, Pz2, 'b')
ax2.set_title('n = 10')
ax3.plot(z, fz, 'k', z, Pz3, 'b')
ax3.set_title('n = 20')

fig.legend(['f(x)', 'P_z(x)'])

for ax in fig.get_axes():
    ax.grid()
    
plt.show()