import matplotlib.pyplot as plt

def plotData(X, y):
    plt.figure()
    plt.scatter(X, y, marker='x', color='red')
    plt.title("Количество автомобилей vs Прибыль СТО")
    plt.xlabel("Количество автомобилей в городе")
    plt.ylabel("Прибыль СТО")
    plt.show()
