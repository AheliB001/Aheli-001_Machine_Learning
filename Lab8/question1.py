import numpy as np

def l2(theta, lambda_):
    total = 0

    for j in range(1, len(theta)):  # skip theta[0]
        total += theta[j] ** 2

    return lambda_ * total


def l1(theta, lambda_):
    total = 0

    for j in range(1, len(theta)):  # skip theta[0]
        total += abs(theta[j])

    return lambda_ * total

theta = np.array([1, 3, -4, 2])
lambda_ = 0.5

print("L1 Norm:", l1(theta, lambda_))
print("L2 Norm:", l2(theta, lambda_))