import numpy as np

H = np.array([
    [-2, 0],
    [0, -6]
])

eigenvalues = np.linalg.eigvals(H)

print("Eigenvalues:")
print(eigenvalues)

if np.all(eigenvalues > 0):
    print("Local Minimum")

elif np.all(eigenvalues < 0):
    print("Local Maximum")

else:
    print("Neither / Saddle Point")