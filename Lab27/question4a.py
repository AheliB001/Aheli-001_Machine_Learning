import sympy as sp

x, y = sp.symbols('x y')
f = 4*x + 2*y - x**2 - 3*y**2

grad_x = sp.diff(f, x)
grad_y = sp.diff(f, y)

print("Gradient:")
print("df/dx =", grad_x)
print("df/dy =", grad_y)

#critical points
critical_points = sp.solve([grad_x, grad_y], (x, y))

print("\nCritical Point:")
print(critical_points)