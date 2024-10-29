import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 2 + 0.75 * np.tanh(2 * x)

def derivative_analytic(x):
    return 1.5 / np.cosh(2 * x)**2  # Derivative of 0.75 * tanh(2x)

def derivative_central(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)


x_vals = np.linspace(-2, 2, 100)


step_sizes = [1, 0.5, 0.1]


plt.figure(figsize=(10, 6))

for h in step_sizes:
    central_diff_vals = np.array([derivative_central(f, x, h) for x in x_vals])
    plt.plot(x_vals, central_diff_vals, label=f'h = {h}')

plt.plot(x_vals, derivative_analytic(x_vals), 'k--', label='Analytic Solution')


plt.title("Central Differencing of f(x) = 2+0.75tanh(2x) vs Analytic Derivative")
plt.xlabel("x")
plt.ylabel("Derivative")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()


