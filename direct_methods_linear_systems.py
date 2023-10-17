import numpy as np


def triang_inf(T, b):
    """
    Solves a system of linear equations where T is a lower triangular matrix.
    Complexity: O(n^2), where n is the number of rows/columns in T
    """
    n = len(b)
    x = np.zeros(n)
    x[0] = b[0]/T[0,0]
    for i in range(1,n):
        S = 0
        for k in range(0,i):
            S += T[i,k]*x[k]
        x[i] = (b[i] - S) / T[i,i]
    return x

def triang_sup(T, b):
    """
    Solves a system of linear equations where T is an upper triangular matrix.
    Complexity: O(n^2), where n is the number of rows/columns in T.
    """
    n = len(b)
    x = np.zeros(n)
    x[n-1] = b[n-1]/T[n-1,n-1]
    for i in range(n-2,-1,-1):
        S = 0
        for k in range(i+1,n):
            S += T[i,k]*x[k]
        x[i] = (b[i] - S) / T[i,i]
    return x

def LU_decomp(A):
    """
    Performs LU decomposition of matrix A.
    Complexity: O(n^3), where n is the number of rows/columns in A. 
    Note: This is a simple implementation. There are more efficient methods available.
    """
    n = len(A)
    L = np.zeros((n,n))
    U = np.zeros((n,n))
    L[0,0] = 1
    for j in range(0,n):
        U[0,j] = A[0,j]
    for i in range(1,n):
        L[i,0] = A[i,0]/U[0,0]
    for i in range(1,n):
        L[i,i] = 1
        S = 0
        for k in range(0,i):
            S += L[i,k]*U[k,i]
        U[i,i] = A[i,i] - S
        for j in range(i+1,n):
            S = 0
            for k in range(0,i):
                S += L[i,k]*U[k,j]
            U[i,j] = A[i,j] - S
            S = 0   
            for k in range(0,i):
                S += L[j,k]*U[k,i]
            L[j,i] = (A[j,i] - S) / U[i,i]
    return L, U

def LU_solve(A, b):
    """
    Solves the system of linear equations Ax = b using LU decomposition.
    Complexity: The decomposition itself is O(n^3) and the solve phase is O(n^2). So, overall it's O(n^3).
    """
    L, U = LU_decomp(A)
    print(L)
    print(U)
    y = triang_inf(L, b)
    x = triang_sup(U, y)
    return x

def cholesky(A):
    """
    Performs the Cholesky decomposition of a symmetric positive-definite matrix A.
    Complexity: O(n^3), where n is the number of rows/columns in A.
    """
    n = len(A)
    C = np.zeros((n,n))
    C[0,0] = np.sqrt(A[0,0])
    for j in range(1,n):
        C[0,j] = A[0,j] / C[0,0]
    for i in range(1,n):
        S = 0
        for k in range(0,i):
            S += C[k,i]**2
        C[i,i] = np.sqrt(A[i,i] - S)
        for j in range(i+1,n):
            S = 0
            for k in range(0,i):
                S += C[k,i]*C[k,j]
            C[i,j] = (A[i,j] - S) / C[i,i]
    return C, C.T

if __name__ == '__main__':
    # Test matrices
    M = np.array([
        [1, 2, 3, 4, 5, 6],
        [7, 8, 9, 10, 11, 12],
        [13, 14, 15, 16, 17, 18],
        [19, 20, 21, 22, 23, 24],
        [25, 26, 27, 28, 29, 30],
        [31, 32, 33, 34, 35, 36]
    ])
    
    M_sdp = np.array([
        [15, 10, 18, 12],
        [10, 15, 7, 13],
        [18, 7, 27, 7],
        [12, 13, 7, 22]
    ])
    
    M2 = np.array([
        [1, 2, 3],
        [1, 4, 3],
        [1, 2, 2]
    ])

    # Test vectors
    b = np.array([8, 2, 3, 2, 5, 6])
    b2 = np.array([14, 13, 9])
    
    # Perform tests
    MTI = np.tril(M)
    MTS = np.triu(M)
    
    print("Matrix M:\n", M)
    print("Lower triangular of M:\n", MTI)
    print("Upper triangular of M:\n", MTS)
    
    print(triang_inf(MTI, b))
    print(triang_sup(MTS, b))
    
    print("Solution using LU Decomposition:\n", LU_solve(M2, b2))
    C, C_t = cholesky(M_sdp)
    print("Result of Cholesky decomposition:\n", np.dot(C, C_t))
