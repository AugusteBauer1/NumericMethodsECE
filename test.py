import numpy as np

def cholesky(A):
    """
    Performs the Cholesky decomposition of a symmetric positive-definite matrix A.
    Complexity: O(n^3), where n is the number of rows/columns in A.
    """
    n = len(A)
    L = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1):
            sum = 0
            if j == i:  # Diagonal elements
                for k in range(j):
                    sum += L[j, k] ** 2
                L[j, j] = np.sqrt(A[j, j] - sum)
            else:
                for k in range(j):
                    sum += L[i, k] * L[j, k]
                if L[j, j] > 0:
                    L[i, j] = (A[i, j] - sum) / L[j, j]
                else:
                    raise ValueError("La matrice n'est pas définie positive.")

    return L

# Exemple d'utilisation
A = np.array([[4, 12, -16],
              [12, 37, -43],
              [-16, -43, 98]], dtype='float64')

L = cholesky(A)
print("L:\n", L)
print("L * L.T:\n", np.dot(L, L.T))
