import numpy as np

def transform(x):
    x1, x2 = x
    return np.array([x1**2, np.sqrt(2)*x1*x2, x2**2])

x1 = np.array([3, 6])
x2 = np.array([10, 10])

#Transform to higher dimension
t1 = transform(x1)
t2 = transform(x2)

#Dot product in higher dimension
dot_product = np.dot(t1, t2)

print("Dot product in higher dimension:", dot_product)