import numpy as np

def ForwardSubRow(L,b):
    """
    Solving Lx = b using forward substitution by rows.
    
    Parameters:
    L : matrix
        n x n lower triangular matrix
    b : vector
        n x 1   
        
    Output
    ------
    x : vector
        n x 1, solution to Lx = b 
    norm : real number
        Residual Max Norm of the method
    """
    n = len(L)
    x = np.zeros(n, dtype=float)
    x[0] = b[0]/L[0][0]
    for i in range(1,n):
        s = 0
        for j in range(0,i):
            s = s + L[i][j]*x[j]
        x[i] = (b[i]-s)/L[i][i]
    return x

def BackwardSubRow(U,b):
    """
    Solving Ux = b using backward substitution by rows.
    
    Parameters:
    U : matrix
        n x n upper triangular matrix
    b : vector
        n x 1   
        
    Output
    ------
    x : vector
        n x 1, solution to Ux = b 
    norm : real number
        Residual Max Norm of the method
    """
    n = len(U)
    x = np.zeros(n, dtype=float)
    x[n-1] = b[n-1]/U[n-1][n-1] 
    for i in range(n-2,-1,-1):
        s = 0
        for j in range(i+1,n):
            s = s + U[i][j]*x[j]
        x[i] = (b[i]-s)/U[i][i]   
    return x 

def LUIJK(A):
    """
    Solving for the LU factorization in A stored in A
    
    Parameters
    ----------
    A : matrix
        n x n
        
    Output
    ------
    A : matrix
        n x n, LU factorization of the input A
    """
    n = len(A)
    for j in range(1,n):
        A[j][0] = A[j][0] / A[0][0]
    for i in range(1,n):
        for j in range(i,n):
            s = 0
            for k in range(0,i):
                s = s + A[i][k]*A[k][j]
            A[i][j] = A[i][j] - s
        for j in range(i+1, n):
            s = 0
            for k in range(0,i):
                s = s + A[j][k]*A[k][i]
            A[j][i] = (A[j][i] - s) / A[i][i]
    return A

def GetLU(A):
    """
    Solving for the LU factors of A
    
    Parameters
    ----------
    A : matrix
        n x n
       
    Output
    ------
    L : lower triangular matrix
    U : upper triangular matrix
    """
    A = LUIJK(A)
    n = len(A)
    L = np.zeros((n,n), dtype=float)
    U = np.zeros((n,n), dtype=float)
    for i in range(0,n):
        L[i][i] = 1
        for j in range(0,i):
            L[i][j] = A[i][j]
        for j in range(i,n):
            U[i][j] = A[i][j]
    return (L, U)
            
def LUSolve(A, b):
    """
    Solving Ax=d using LU Factorization

    Parameters
    ----------
    A : matrix
        n x n
    b : vector
        n x 1
        
    Output
    ------
    x : vector
        Solution to Ax=d
    normx : real number
        Residual Max Norm of the method
    """
    L, U = GetLU(A)
    y = ForwardSubRow(L, b)
    x = BackwardSubRow(U, y)
    return x