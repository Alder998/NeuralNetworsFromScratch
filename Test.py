import NNotebook as NN
import pandas as pd
import numpy as np
import random
import NNLibrary

X1, X2, X3, X4, X5, XR = list(), list(), list(), list(), list(), list()
columns = ['X1', 'X2', 'X3', 'X4', 'X5', 'Pred']
cR = ['X1', 'X2', 'X3', 'X4', 'X5']

lines = 50000
for line in range(lines):
    X1.append(random.random())
    X2.append(random.random() * random.choice([-2, 2]))
    X3.append(random.random() * random.choice([-0.5, 0.5]))
    X4.append(random.random() * random.choice([-3, 3]))
    X5.append(random.random() * random.choice([-23, 23]))
    XR.append(random.random() * random.choice([-10, 10]))

simulatedData = pd.concat([pd.Series(X1), pd.Series(X2), pd.Series(X3), pd.Series(X4),
                           pd.Series(X5), np.abs(pd.Series(XR))], axis=1).set_axis(columns,
                                                                                   axis=1)

# get Structure

NNt = NNLibrary.NeuralNetwork(6, 1, 9)
NNFit = NNt.fit(simulatedData, 'Pred', 0.005)





