import numpy as np
import time as t

def triang_inf(T, b):
    """
    Solves a system of linear equations where T is a lower triangular matrix.
    Complexity: O(n^2), where n is the number of rows/columns in T.
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

def inv_diag(D):
    """
    Inverts a diagonal matrix.
    Complexity: O(n), where n is the number of rows/columns in D.
    """
    n = len(D)
    Dinv = np.zeros((n,n))
    for i in range(0,n):
        Dinv[i,i] = 1/D[i,i]
    return Dinv

def gauss_seidel(A,b,x0,epsilon):
    """
    Implements the Gauss-Seidel iterative method to solve Ax = b.
    Complexity: Iterative, depends on the convergence criteria (epsilon). In each iteration, complexity is O(n^2).
    """

    # Define D, M and N as A = D - M - N, with M strictly lower triangular and N strictly upper triangular
    D = np.diag(np.diag(A))
    M = -np.tril(A,-1)
    N = -np.triu(A,1)
    print("D = \n", D)
    print("M = \n", M)
    print("N = \n", N)

    error = 3*epsilon
    count = 0

    while error > epsilon:
        count += 1
        x = triang_inf(D-M, b+np.dot(N,x0))
        error = np.linalg.norm(x-x0)
        x0 = x
    return x, count

def jacobi(A,b,x0,epsilon):
    """
    Implements the Jacobi iterative method to solve Ax = b.
    Complexity: Iterative, depends on the convergence criteria (epsilon). In each iteration, complexity is O(n^2).
    """

    # Define D, M and N as A = D - M - N, with M strictly lower triangular and N strictly upper triangular
    D = np.diag(np.diag(A))
    M = -np.tril(A,-1)
    N = -np.triu(A,1)
    print("D = \n", D)
    print("M = \n", M)
    print("N = \n", N)

    error = 10*epsilon

    count = 0
    while error > epsilon:
        Dinv = inv_diag(D)
        x = np.dot(Dinv,M+N).dot(x0) + np.dot(Dinv,b)
        error = np.linalg.norm(x-x0)
        x0 = x
        count += 1
    return x, count

def SOR(A,b,x0,epsilon,omega):
    """
    Implements the Successive Over-relaxation (SOR) method to solve Ax = b.
    Complexity: Iterative, depends on the convergence criteria (epsilon). In each iteration, complexity is O(n^2).
    """

    # Define D, M and N as A = D - M - N, with M strictly lower triangular and N strictly upper triangular
    D = np.diag(np.diag(A))
    M = -np.tril(A,-1)
    N = -np.triu(A,1)
    print("D = \n", D)
    print("M = \n", M)
    print("N = \n", N)

    error = 10*epsilon
    count = 0
    while error > epsilon:
        x = triang_inf(D - omega * M, omega * b + (1-omega) * np.dot(D, x0) + omega * np.dot(N,x0))
        error = np.linalg.norm(x-x0)
        x0 = x
        count += 1
    return x, count


def generate_tridiagonal(N):
    """
    Generates a tridiagonal matrix of size N with main diagonal having value 3 and sub/super diagonals having value 1.
    Also generates a vector b of size N with all values set to 1.
    Complexity: O(n), where n is the size of the matrix.
    """

    A = np.zeros((N,N))
    b = np.ones(N)
    for i in range(0,N):
        A[i,i] = 3
        if i != N-1:
            A[i,i+1] = 1
            A[i+1,i] = 1
    return A, b



if __name__ == '__main__':
    # Generate test matrices and vectors
    A, b = generate_tridiagonal(1000)
    A2 = np.array([[2,1,0], [1,2,1], [0,1,2]])
    b2 = np.array([3,-1,1])

    # Test and time Gauss-Seidel method
    start_time = t.time()
    print(gauss_seidel(A, b, np.zeros(len(A)), 0.0001))
    end_time = t.time()
    print("Execution time for Gauss Seidel:", end_time - start_time)

    # Test and time Jacobi method
    start_time = t.time()
    print(jacobi(A, b, np.zeros(len(A)), 0.0001))
    end_time = t.time()
    print("Execution time for Jacobi:", end_time - start_time)

    # Test and time SOR method
    start_time = t.time()
    print(SOR(A, b, np.zeros(len(A)), 0.0001, 0.9))
    end_time = t.time()
    print("Execution time for SOR:", end_time - start_time)

    # Test SOR with a different matrix
    print(SOR(A2, b2, np.zeros(len(A2)), 0.0001, 1.5))