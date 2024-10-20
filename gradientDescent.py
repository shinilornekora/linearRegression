from computeCost import computeCost


def gradientDescent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []

    for _ in range(iterations):
        temp_theta = [0, 0]
        
        for j in range(2):
            summation = 0
            for i in range(m):
                prediction = theta[0] + theta[1] * X[i][1]
                summation += (prediction - y[i]) * (X[i][j] if j == 1 else 1)
            temp_theta[j] = theta[j] - (alpha / m) * summation
        
        theta = temp_theta
        cost_history.append(computeCost(X, y, theta))

    return theta, cost_history
