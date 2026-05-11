import numpy as np

A = np.array([
    [9, -15],
    [-15, 21]
])

eigenvalues = np.linalg.eigvals(A)

print("Eigenvalues:", eigenvalues)

if np.all(eigenvalues > 0):
    print("Positive Definite")
else:
    print("Not Positive Definite")