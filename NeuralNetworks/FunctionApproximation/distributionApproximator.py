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

# Questo esperimento è strutturato cosi:
# - Prendere tutti i rendimenti di una azione quotata
# - Calcolarne la Empirical Cumulative Distribution Function
# - Dare i rendimenti in pasto a una rete neurale modificata, che ha ultimo layer di lunghezza n (n=numero
# di parametri da stimare)
# - Una volta stimati questi parametri per ognuno dei redimenti, farne la media per avere un valore univoco
# - Creare una mixture distribution con questi parametri
# - Calcolare un K-S Test di questa distribuzione risultante con quella di Input
# - Iterare finchè il D-Statistic del test non aumenta.

# Prendiamo i dati, e li puliamo

SData = (yf.Ticker('LIN').history('20Y')['Close'].pct_change()*100).dropna().reset_index()
del[SData['Date']]
SData = SData.set_axis(['ret'], axis = 1)

# Creiamo la nostra mixture con parametri simulati

SData = pd.concat([SData['ret'], pd.DataFrame(np.full(len(SData['ret']),
            np.random.uniform(size = len(SData['ret'])))).set_axis(['random'], axis = 1) ], axis = 1)
SData['random'] = SData['random']

# Creiamo la rete

structure = [100]

NNt = NeuralNetworks.Regression(2, structure)
NNFit = NNt.fit(SData, 'ret', 0.008, decreasingRate = 0.99)

prediction = NNt.getPredictions2(NNFit, SData, 'ret')
#print(prediction)

# Plottiamo

plt.figure(figsize=(12, 4))
plt.plot(SData['ret'], color = 'blue')
plt.plot(prediction, color = 'red')
plt.show()

# Creazione della Empirical Cumulative Distribution

ret = SData['ret']
e_cum = ECDF(ret)
sample = np.linspace(ret.min(), ret.max(), len(ret))
cum = list()
for i in sample:
    cum.append(e_cum(i))
cum = pd.Series(cum)

retR = prediction
e_cumR = ECDF(retR)
sampleR = np.linspace(retR.min(), retR.max(), len(retR))
cumR = list()
for i in sampleR:
    cumR.append(e_cumR(i))
cumR = pd.Series(cumR)

plt.figure(figsize=(12, 4))
plt.plot(cum, color = 'blue')
plt.plot(cumR, color = 'red')
plt.show()

cumulativeR = pd.concat([cum, cumR], axis = 1).set_axis(['returns', 'random'], axis = 1)



