import matplotlib.pyplot as plt
from plotData import plotData
from computeCost import computeCost_elementwise, computeCost_vectorized
from gradientDescent import gradientDescent_elementwise, gradientDescent_vectorized

data = []
with open('ex1data1.txt', 'r') as file:
    for line in file:
        data.append([float(num) for num in line.split(',')])

X = [row[0] for row in data]
y = [row[1] for row in data]
m = len(y)
X = [[1, x] for x in X]
theta = [0, 0]
iterations = 1500
alpha = 0.01

plotData([x[1] for x in X], y)

variant = 'elementwise'

if variant == 'elementwise':
    initial_cost = computeCost_elementwise(X, y, theta)
    theta, cost_history = gradientDescent_elementwise(X, y, theta, alpha, iterations)
else:
    initial_cost = computeCost_vectorized(X, y, theta)
    theta, cost_history = gradientDescent_vectorized(X, y, theta, alpha, iterations)

print(f"Начальная стоимость: {initial_cost}")
print(f"Оптимизированные параметры theta: {theta}")

plt.plot([x[1] for x in X], [theta[0] + theta[1] * x[1] for x in X], label='Линейная регрессия', color='red')
plt.legend()
plt.show()

theta0_vals = [t / 100 for t in range(-1000, 1000, 50)]
theta1_vals = [t / 100 for t in range(-1000, 1000, 50)]
J_vals = []

for t0 in theta0_vals:
    row = []
    for t1 in theta1_vals:
        t = [t0, t1]
        if variant == 'elementwise':
            cost = computeCost_elementwise(X, y, t)
        else:
            cost = computeCost_vectorized(X, y, t)
        row.append(cost)
    J_vals.append(row)

theta0_vals_grid = []
theta1_vals_grid = []

for t0 in theta0_vals:
    for t1 in theta1_vals:
        theta0_vals_grid.append(t0)
        theta1_vals_grid.append(t1)

J_vals_flat = [J_vals[i // len(theta1_vals)][i % len(theta1_vals)] for i in range(len(theta0_vals) * len(theta1_vals))]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(theta0_vals_grid, theta1_vals_grid, J_vals_flat, cmap='viridis')

ax.set_xlabel('theta_0')
ax.set_ylabel('theta_1')
ax.set_zlabel('Cost J')

plt.show()
