import numpy as np
import linearsystem as ls

def f(x):
    """
    Vector-valued function

    Args:
        x : vector
        
    Output:
        f : vector
            function value at x
    """
    # 20ac − 8ab + 16c^3 − 4bd = 39
    f1 = 20*x[0]*x[2] - 8*x[0]*x[1] + 16*x[2]**3 - 4*x[1]*x[3] - 39
    # 12a + 6b + 2c − 2d = 11
    f2 = 12*x[0] + 6*x[1] + 2*x[2] - 2*x[3] - 11
    # 4a^2 + 2bc − 10c + 2ad^2 = −7
    f3 = 4*x[0]**2 + 2*x[1]*x[2] - 10*x[2] + 2*x[0]*x[3]**2 + 7
    # −3ad − 2b^2 + 7cd = 16
    f4 = -3*x[0]*x[3] - 2*x[1]**2 + 7*x[2]*x[3] - 16
    f = np.array([f1,f2,f3,f4])
    return f

def Jf(x):
    """
    Jacobian of f
    
    Args:
        x : vector
        
    Output:
        Jf : matrix
            Jacobian of f at x
    """
    Jf0n = [20*x[2]-8*x[1], -8*x[0]-4*x[3], 20*x[0]+48*x[2]**2, -4*x[1]]
    Jf1n = [12, 6, 2, -2]
    Jf2n = [8*x[0]+2*x[3]**2, 2*x[2], 2*x[1]-10, 4*x[0]*x[3]]
    Jf3n = [-3*x[3], -4*x[1], 7*x[3], -3*x[0]+7*x[2]]
    Jf = np.array([Jf0n, Jf1n, Jf2n, Jf3n])
    return Jf

def NewtonSystem(f, Jf, x, tol, maxit):
    """
    Solves Nonlinear System f(x)=0 using Newton's Method

    Args:
        f : callable vector-valued function, outputs vector
            Nonlinear Function
        Jf : callable vector-valued function, outputs matrix
            Jacobian of f at x
        x : vector
            Initial iterate
        tol : scalar
            Tolerance
        maxit : integer scalar
            Maximum iteration number
            
    Output:
        x : vector
            Solution to f(x) = 0
        fx : vector
            Function value at x
        k : integer scalar
            Number of iterates
    """
    k = 0
    err = tol + 1
    while (err > tol) & (k < maxit):
        dx = ls.LUSolve(Jf(x),-f(x))
        x = x + dx
        # err = np.linalg.norm(dx,np.inf)
        err = max(row.sum() for row in np.abs(dx))
        k += 1
    if (err > tol) & (k == maxit):
        print("Method Fails")
    return x, f(x), err, k

# parameters
initx = [1, 1, 1, 1]
tol = 1e-15
maxit = 100

# float_formatter = "{:.15f}".format
# np.set_printoptions(formatter={'float_kind':float_formatter})
x, fx, err, k = NewtonSystem(f, Jf, initx, tol, maxit)
print("x = ", x)
print("f(x) = ", fx)
print("err = ", err)
print("NumOfIter = ", k)