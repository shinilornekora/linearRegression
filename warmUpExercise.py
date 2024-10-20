def warmUpExercise_elementwise(n):
    I_custom = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    return I_custom

def warmUpExercise_vectorized(n):
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]
