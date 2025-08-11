import numpy as np 

trainExamples = [
        (1,1),
        (2,3),
        (4,3),
]

def phi(x):
    return np.array([1, x])

def initialWeightVector():
    return np.array([0, 0])
def TrainingLoss(w):
    return 1.0/len(trainExamples) * sum((w.dot(phi(x) - y))** 2 for x, y in trainExamples)

def gradientTrainLoss(w):
    return 1.0/len(trainExamples) * sum(2 * (w.dot(phi(x)) - y) * phi(x) for x, y in trainExamples)
def gradientDescent(F, gradientF, initialWeightVector):
    w = initialWeightVector()
    eta=0.1
    for t in range(500): 
        value=F(w) 
        gradient = gradientF(w)
        w = w - eta*gradient
        print(f'epoch{t}: w={w},F(w)={value}, gradientF={gradient}')
        
gradientDescent(TrainingLoss, gradientTrainLoss, initialWeightVector)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
def plotTrainingData():
    x = [x for x, y in trainExamples]
    y = [y for x, y in trainExamples]
    plt.scatter(x, y, color='blue', label='Training Data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Training Data')
    plt.legend()
def plotRegressionLine(w):
    x = np.linspace(0, 5, 100)
    y = w[0] + w[1] * x
    plt.plot(x, y, color='red', label='Regression Line')
    plt.legend()
plotTrainingData()
plotRegressionLine(initialWeightVector())
plt.show()
plotRegressionLine(np.array([0.5, 0.5]))  # Example of plotting with a different weight vector
plt.show()
plotRegressionLine(np.array([1, 1]))  # Example of plotting with a different weight vector
plt.show()
plotRegressionLine(np.array([2, 1]))
plt.show()
plotRegressionLine(np.array([3, 0.5]))
plt.show()


#--------------------------------------------------------------------------

import numpy as np 

# Training dataset: (x, y) pairs
trainExamples = [
    (1, 1),
    (2, 3),
    (4, 3),
]

# Feature mapping: adds bias term 1
def phi(x):
    return np.array([1, x])

# Initial weights (bias = 0, slope = 0)
def initialWeightVector():
    return np.array([0.0, 0.0])

# Mean Squared Error cost function
def TrainingLoss(w):
    return (1.0 / len(trainExamples)) * sum(
        (w.dot(phi(x)) - y) ** 2 for x, y in trainExamples
    )

# Gradient of the cost function
def gradientTrainLoss(w):
    return (1.0 / len(trainExamples)) * sum(
        2 * (w.dot(phi(x)) - y) * phi(x) for x, y in trainExamples
    )

# Gradient Descent loop
def gradientDescent(F, gradientF, initialWeightVector):
    w = initialWeightVector()
    eta = 0.1  # learning rate
    for t in range(20):  # only show first 20 steps for clarity
        value = F(w)
        gradient = gradientF(w)
        w = w - eta * gradient
        print(f'Epoch {t:02d}: w={w}, F(w)={value:.4f}, gradient={gradient}')
    return w

# Run gradient descent
final_weights = gradientDescent(TrainingLoss, gradientTrainLoss, initialWeightVector)
print("\nFinal weights:", final_weights)

