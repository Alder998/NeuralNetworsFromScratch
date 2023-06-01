import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from NeuralNetworks import NeuralNetworks

dataMeteo = pd.read_excel(r"C:\Users\39328\OneDrive\Desktop\Davide\Velleità\Meteo_Data.xlsx")
dataMeteo = dataMeteo[['elevation', 'Month', 'lat', 'lng', 'Average Temp']].set_axis(['X1',
                                                                    'X2', 'X3', 'X4', 'Pred'], axis = 1)

df = train_test_split(dataMeteo, test_size=0.20)

trainSet = df[0]
testSet = df[1]

print('\n')
print('Train Size:', len(trainSet['X1']))
print('\n')

# get Structure

structure = [10]

NNt = NeuralNetworks.Regression(5, structure)
NNFit = NNt.fit(trainSet, 'Pred', 0.02, decreasingRate = 0.99)

prediction = NNt.getPredictions2(NNFit, testSet, 'Pred')

# Regressione Lineare

X = np.array(trainSet[['X1', 'X2', 'X3', 'X4']])
Y = np.array(trainSet['Pred'])

reg = LinearRegression().fit(X, Y)

f = reg.predict(np.array(testSet[['X1', 'X2', 'X3', 'X4']]))

# Mettiamo insieme

final = pd.concat([prediction, pd.Series(f), testSet['Pred']], axis = 1).set_axis(['Predicted NN',
                                                            'Predicted Linear Regression', 'Real'], axis = 1)
# Calcoliamo il MSE sul test

MSERegression = ((final['Predicted Linear Regression'] - final['Real'])**2).mean()
MSENN = ((final['Predicted NN'] - final['Real'])**2).mean()

print('\n')
print('TEST SET MSE')
print('\n')
print('MSE Neural Network', MSENN)
print('MSE Linear Regression', MSERegression)

# R-Squared fatto in casa per la rete neurale. Ricorda che l'R-squared è la parte di volatilità che il modello
# "spiega" sulla volatilità totale. Quindi, in questo caso: Sigma(model Prediction) / Sigma(True values)

#NNRSquared = (final['Predicted NN'].std() / final['Real'].std())
#LRScore = reg.score(np.array(testSet[['X1', 'X2', 'X3', 'X4']]), np.array(testSet['Pred']))
#
#print('\n')
#print('R-Squared Neural Network', round(NNRSquared*100, 2), '%')
#print('R-Squared Linear Regression', round(LRScore*100, 2), '%')




