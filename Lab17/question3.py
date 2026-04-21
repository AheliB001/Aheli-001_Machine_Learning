import numpy as np

def polynomial_kernel(a,b):
    return (a[0]**2)*(b[0]**2) + 2*a[0]*b[0]*a[1]*b[1] + (a[1]**2)*(b[1]**2)

x1=np.array([3,6])
x2=np.array([10,10])

kernel=polynomial_kernel(x1,x2)
print(kernel)

#transform plus dot product and kernel alone gives the same output.
#Kernel trick allows us to compute inner products in higher dimensions without explicitly transforming the data

