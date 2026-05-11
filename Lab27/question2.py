import numpy as np

#Hessian matrix at (3,1)
H = np.array([
    [12*(3**2), -1],
    [-1, 2]
])

#eigenvalues
eigenvalues = np.linalg.eigvals(H)

print("Eigenvalues:")
print(eigenvalues)