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

classes = 10

# Generiamo dei dati Random

dataMeteo = pd.read_excel(r"C:\Users\39328\OneDrive\Desktop\Davide\Velleità\Meteo_Data_simplified.xlsx")
dataMeteo = dataMeteo[['elevation', 'Month', 'lat', 'lng', 'Description']].set_axis(['X1',
                                                                    'X2', 'X3', 'X4', 'Pred'], axis = 1)

# Encoding della variabile categorica

for i, singleValue in enumerate(dataMeteo['Pred'].unique()):
    dataMeteo.loc[dataMeteo['Pred'] == singleValue, 'Pred_enc'] = i + 1

dataMeteo['Pred_enc'] = dataMeteo['Pred_enc'].astype(int)
del[dataMeteo['Pred']]

print('Categories:', len(dataMeteo['Pred_enc'].unique()))

df = train_test_split(dataMeteo, test_size=0.20)

dataMeteoTrain = df[0]
dataMeteoTest = df[1]

print('\n')
print('Train Size:', len(dataMeteoTrain['X1']))
print('\n')


# get Structure

structure = [10]

NNt = NeuralNetworks.Classification(5, structure, classes)
NNFit = NNt.fit(dataMeteoTrain, 'Pred_enc', 0.04, decreasingRate = 0.99, max_iteration_epochs=10, max_iteration_batch=5)

prediction = NNt.getPredictionsC(NNFit, dataMeteoTrain, 'Pred_enc')
