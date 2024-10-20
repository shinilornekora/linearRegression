from computeCost import computeCost_elementwise, computeCost_vectorized

def gradientDescent_elementwise(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []

    for _ in range(iterations):
        temp_theta = [0, 0]
        
        for j in range(2):
            summation = 0
            for i in range(m):
                prediction = theta[0] + theta[1] * X[i][1]
                error = (prediction - y[i]) * (X[i][j] if j == 1 else 1)
                summation += error
            temp_theta[j] = theta[j] - (alpha / m) * summation
        
        theta = temp_theta
        cost_history.append(computeCost_elementwise(X, y, theta))

    return theta, cost_history


def gradientDescent_vectorized(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []

    for _ in range(iterations):
        predictions = [theta[0] + theta[1] * X[i][1] for i in range(m)]
        errors = [predictions[i] - y[i] for i in range(m)]
        theta[0] = theta[0] - (alpha / m) * sum(errors)
        theta[1] = theta[1] - (alpha / m) * sum(errors[i] * X[i][1] for i in range(m))
        cost_history.append(computeCost_vectorized(X, y, theta))

    return theta, cost_history
