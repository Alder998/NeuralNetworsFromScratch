# Qui lavoriamo per l'approssimazione delle distribuzioni. Il progetto non è semplice, dal momento che cercheremo di
# lavorare sull'approssimazione della distribuzione dei rendimenti di alcune azioni quotate (perchè ci sono tanti dati)

import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from NeuralNetworks import NeuralNetworks

# crea dei punti random nello spazio

nPoints = 50
data = pd.concat([pd.DataFrame(np.full(50, np.random.normal(loc = 1, scale = 0.03, size = nPoints))).set_axis(['target'], axis = 1),
                  pd.DataFrame(np.full(50, np.random.normal(loc = 1, scale = 1, size = nPoints))).set_axis(['start'], axis = 1) ], axis = 1)
print(data)

# Creiamo la rete

structure = [100]

NNt = NeuralNetworks.Regression(2, structure)
NNFit = NNt.fit(data, 'target', 0.01, decreasingRate = 0.99)

prediction = NNt.getPredictions2(NNFit, data, 'target')
print(pd.concat([data['target'], prediction], axis = 1).set_axis(['target','predicted'], axis = 1))

# Plottiamo

plt.figure(figsize=(12, 6))

plt.subplot(2,1,1)
plt.scatter(y = data['target'], x = data.index, color = 'red')
plt.plot(data['start'], color = 'blue')

plt.subplot(2,1,2)
plt.scatter(y = data['target'], x = data.index, color = 'red')
plt.plot(prediction, color = 'blue')

plt.show()
