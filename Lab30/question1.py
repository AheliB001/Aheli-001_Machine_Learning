import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

# Simulate 1000 points from N(10,3)
data = np.random.normal(10, 3, 1000)

# Negative log-likelihood
def nll(params):
    mu, sigma = params
    return -np.sum(norm.logpdf(data, mu, sigma))

# Optimize
result = minimize(nll, [1, 1])

# Estimated parameters
print("Estimated mu =", result.x[0])
print("Estimated sigma =", result.x[1])