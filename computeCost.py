def computeCost_elementwise(X, y, theta):
    m = len(y)
    cost = 0

    for i in range(m):
        prediction = theta[0] + theta[1] * X[i][1]
        error = prediction - y[i]
        cost += error ** 2

    return cost / (2 * m)


def computeCost_vectorized(X, y, theta):
    m = len(y)

    predictions = [theta[0] + theta[1] * X[i][1] for i in range(m)]
    errors = [predictions[i] - y[i] for i in range(m)]
    squared_errors = [error ** 2 for error in errors]
    cost = sum(squared_errors) / (2 * m)
    
    return cost
