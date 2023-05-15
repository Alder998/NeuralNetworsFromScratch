import NNotebook as NN
import pandas as pd
import numpy as np
import random
import NNLibrary

dataMeteo = pd.read_excel(r"C:\Users\39328\OneDrive\Desktop\Davide\Velleità\Meteo_Italia.xlsx")
dataMeteo = dataMeteo[['Wind Speed (km/h)', 'Humidity (%)', 'Average Temp']].set_axis(['X1', 'X2', 'Pred'],
                                                                                      axis = 1)

# get Structure

NNt = NNLibrary.NeuralNetwork(3, 1, 9)
NNFit = NNt.fit(dataMeteo, 'Pred', 0.005)





