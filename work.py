import numpy as np

def predict_profit(num_cars, theta):
    X_input = np.array([1, num_cars])
    profit = X_input.dot(theta)
    
    return profit
