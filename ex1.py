import matplotlib.pyplot as plt
from warmUpExercise import warmUpExercise
from plotData import plotData
from computeCost import computeCost
from gradientDescent import gradientDescent

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
initial_cost = computeCost(X, y, theta)

print(f"Начальная стоимость: {initial_cost}")
theta, cost_history = gradientDescent(X, y, theta, alpha, iterations)
print(f"Оптимизированные параметры theta: {theta}")

plt.plot([x[1] for x in X], [theta[0] + theta[1] * x[1] for x in X], label='Линейная регрессия', color='red')
plt.legend()
plt.show()

theta0_vals = [-10 + 0.2 * i for i in range(100)]
theta1_vals = [-1 + 0.05 * i for i in range(100)]

J_vals = [[computeCost(X, y, [theta0, theta1]) for theta1 in theta1_vals] for theta0 in theta0_vals]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

Theta0, Theta1 = [], []
for t0 in theta0_vals:
    for t1 in theta1_vals:
        Theta0.append(t0)
        Theta1.append(t1)

ax.plot_trisurf(Theta0, Theta1, [computeCost(X, y, [t0, t1]) for t0 in theta0_vals for t1 in theta1_vals], cmap='viridis')

ax.set_xlabel('theta_0')
ax.set_ylabel('theta_1')
ax.set_zlabel('Функция стоимости J(theta)')
ax.set_title('3D-график функции стоимости')

plt.show()
