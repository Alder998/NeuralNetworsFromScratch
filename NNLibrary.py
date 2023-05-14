# Questa libreria serve per dare un design OOP ai metodi per creare una rete neurale
# Questo serve anche per creare nuovi Spinoff di reti, come CNN, o, molto più probabilmente, RNN.

# Allo stesso tempo questo serve anche per Fare in modo che la rete neurale approcci più tipi di
# problemi (regressione, classificazione, classificazione multipla).

# Classe inizializzazione

class NeuralNetwork:
    name = "NN From Scratch"

    def __init__(self, inputLayer, numberOfLayers, numberOfNodes):
        self.numberOfLayers = numberOfLayers
        self.numberOfNodes = numberOfNodes
        self.inputLayer = inputLayer

    # Metodo per ottenere la struttura della Rete Neurale

    def structure (self):

        import pandas as pd
        import numpy as np
        import random

        numberOfLayers = self.numberOfLayers
        numberOfNodes = self.numberOfNodes

        # Costruiamo i pesi

        weights = list()
        beta = list()
        interceptsW = list()
        neuralIntercept = ['B0']
        for layer in range(1, numberOfLayers + 1):

            # Definiamo le operazioni da svolgere nel singolo nodo

            for node in range(1, numberOfNodes + 1):

                b = 'B' + str(node)
                w0 = 'w0' + str(node)

                beta.append(b)
                interceptsW.append(w0)

                for parameter in range(1, self.inputLayer):
                    w = 'W' + str(node) + str(parameter)

                    weights.append(w)

                    # All'interno di ogni singolo nodo, infatti, c'è l'equazione di una regressione lineare, che
                    # Definiamo come Ak

        # print('Total Number of Parameters to estimate:', len(beta) + len(interceptsW) + len(weights) + 1)
        # print('\n')

        wRandom = pd.concat([pd.Series(weights), pd.Series(np.random.uniform(size=len(weights)))],
                            axis=1).set_axis(['parameter', 'value'], axis=1)
        interceptRandom = pd.concat([pd.Series(interceptsW), pd.Series(np.random.uniform(size=len(interceptsW)))],
                                    axis=1).set_axis(['parameter', 'value'], axis=1)
        betaRandom = pd.concat([pd.Series(beta), pd.Series(np.random.uniform(size=len(beta)))],
                               axis=1).set_axis(['parameter', 'value'], axis=1)
        functionIntRandom = pd.concat([pd.Series(neuralIntercept), pd.Series(random.random() * random.choice([-1, 1]))],
                                      axis=1).set_axis(['parameter', 'value'], axis=1)

        # L'output va interpretato così:
        # res[0] = Beta che vanno inseriti nella f(X) finale (numero nodi)
        # res[1] = Intercette per ogni nodo (numero nodi)
        # res[2] = pesi per ogni nodo (numero nodi x numero predictors)
        # res[3] = intercetta della rete (un solo valore)

        return pd.concat([betaRandom, interceptRandom, wRandom, functionIntRandom], axis=0)
