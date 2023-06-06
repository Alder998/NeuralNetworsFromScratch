import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from NeuralNetworks import NeuralNetworks
import random

#dataMeteo = pd.read_excel(r"C:\Users\39328\OneDrive\Desktop\Davide\Velleità\Meteo_Data.xlsx")
#dataMeteo = dataMeteo[['elevation', 'Month', 'lat', 'lng', 'Average Temp']].set_axis(['X1',
#                                                                    'X2', 'X3', 'X4', 'Pred'], axis = 1)

# dati simulati a questo giro

classes = 9

# Generiamo dei dati Random

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
    XR.append(random.choice(np.arange(1, classes)))

simulatedData = pd.concat([pd.Series(X1), pd.Series(X2), pd.Series(X3), pd.Series(X4),
                           pd.Series(X5), np.abs(pd.Series(XR))], axis=1).set_axis(columns,
                                                                                   axis=1)

df = train_test_split(simulatedData, test_size=0.20)

trainSet = df[0]
testSet = df[1]

print('\n')
print('Train Size:', len(trainSet['X1']))
print('\n')

# get Structure

structure = [10, 5]

NNt = NeuralNetworks.Classification(6, structure, classes)
NNFit = NNt.fit(trainSet, 'Pred', 0.10, decreasingRate = 0.99)

prediction = NNt.getPredictionsC(NNFit, testSet, 'Pred')
