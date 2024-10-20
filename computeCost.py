def computeCost(X, y, theta):
    m = len(y)
    cost = 0
    
    for i in range(m):
        prediction = theta[0] + theta[1] * X[i][1]
        error = prediction - y[i]
        cost += error ** 2 
    
    return cost / (2 * m)
