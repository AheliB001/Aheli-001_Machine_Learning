import numpy as np

def check_point(x, y):

    H = np.array([
        [6*x, -1],
        [-1, 12*y]
    ])

    eigenvalues = np.linalg.eigvals(H)

    print(f"\nPoint ({x},{y})")
    print("Hessian Matrix:")
    print(H)

    print("Eigenvalues:")
    print(eigenvalues)

    if np.all(eigenvalues > 0):
        print("Local Minimum")

    elif np.all(eigenvalues < 0):
        print("Local Maximum")

    else:
        print("Saddle Point")


#points
check_point(0, 0)

check_point(3, 3)

check_point(3, -3)