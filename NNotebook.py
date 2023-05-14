def getParameterVector(data, numberOfLayers, numberOfNodes):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import random

    numberOfLayers = numberOfLayers
    numberOfNodes = numberOfNodes

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

            for parameter in range(1, len(data.columns)):
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


def getPredictions (parameterVector, data, dependent, numberOfLayers, numberOfNodes):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import random

    b = parameterVector

    wRandom = np.array(b['value'][b['parameter'].str.contains('W')])
    interceptRandom = np.array(b['value'][b['parameter'].str.contains('w0')])
    betaRandom = np.array(b['value'][(b['parameter'].str.contains('B')) & (~b['parameter'].str.contains('B0'))])
    functionIntRandom = np.array(b['value'][b['parameter'].str.contains('B0')])

    # Costruiamo ogni singolo Nodo usando i dati random e una funzione non lineare reLu

    functionListNode = list()
    nodeO = 0
    while nodeO < numberOfNodes:
        regrWeights = (wRandom[nodeO:nodeO + len(data.columns) - 1])
        singleFunction = (interceptRandom[nodeO] + (regrWeights * data.loc[:, data.columns != dependent]))
        reLuFunction = pd.DataFrame(np.where(singleFunction > 0, singleFunction, 0)).set_axis(
            data.loc[:, data.columns != dependent].columns, axis=1)

        functionListNode.append(reLuFunction)

        nodeO += 1

    # Abbiamo costruito ogni singolo nodo, ora dobbiamo inserirlo nell'equazione generale

    predList = list()
    for SingleNodeValues in range(len(functionListNode)):
        predList.append(betaRandom[SingleNodeValues] * functionListNode[SingleNodeValues])

    equationResult = list()
    for predictorEquationsResults in predList:
        equationResult.append((functionIntRandom[0] + predictorEquationsResults).transpose().sum().transpose())

    equationResult = pd.concat([series for series in equationResult], axis=1)
    equationResult = equationResult.transpose().sum().transpose()

    return equationResult


# funzione per il calcolo del gradiente (La per le reti neurali standard la formula è sempre uguale, quindi per ogni
# Variabile sottostante il gradiente si calcola nello stesso modo)

def computeGradient(variable, data, numberOfLayers, numberOfNodes):

    import pandas as pd
    import numpy as np

    b = getParameterVector(data, numberOfLayers, numberOfNodes)

    if variable == 'B0':
        return np.full(500000, 1).mean()

    if (variable[0] == 'B') & (variable != 'B0'):
        wRandom = np.array(b['value'][(b['parameter'].str.contains('W')) & (b['parameter'].str.contains(variable[1]))])
        wRandom = wRandom[0:len(data.loc[:, data.columns != 'Pred'].columns)]

        interceptRandom = np.array(b['value'][(b['parameter'].str.contains('w0')) &
                                              (b['parameter'].str.contains(variable[1]))])

        betaRandom = np.array(b['value'][(b['parameter'].str.contains('B')) & (~b['parameter'].str.contains('B0'))])
        functionIntRandom = np.array(b['value'][b['parameter'].str.contains('B0')])

        # Costruiamo ogni singolo Nodo usando i dati random e una funzione non lineare reLu

        singleFunction = (interceptRandom[0] + (wRandom * data.loc[:, data.columns != 'Pred']))
        reLuFunction = pd.DataFrame(np.where(singleFunction > 0, singleFunction, 0)).set_axis(
            data.loc[:, data.columns != 'Pred'].columns, axis=1)

        return (np.array(reLuFunction)).flatten().mean()

    if (variable[0] == 'w'):
        wRandom = np.array(b['value'][(b['parameter'].str.contains('W')) & (b['parameter'].str.contains(variable[2]))])
        wRandom = wRandom[0:len(data.loc[:, data.columns != 'Pred'].columns)]

        interceptRandom = np.array(b['value'][(b['parameter'].str.contains('w0')) &
                                              (b['parameter'].str.contains(variable[2]))])

        betaRandom = np.array(b['value'][(b['parameter'].str.contains('B')) & (~b['parameter'].str.contains('B0'))])
        functionIntRandom = np.array(b['value'][b['parameter'].str.contains('B0')])

        singleFunction = (interceptRandom[0] + (wRandom * data.loc[:, data.columns != 'Pred']))
        reLuFunction = pd.DataFrame(np.where(singleFunction > 0, 1, 0)).set_axis(
            data.loc[:, data.columns != 'Pred'].columns, axis=1)

        return (np.array(reLuFunction)).flatten().mean()

    if (variable[0] == 'W'):
        wRandom = np.array(b['value'][(b['parameter'].str.contains('W')) & (b['parameter'].str.contains(variable[1]))])
        wRandom = wRandom[0:len(data.loc[:, data.columns != 'Pred'].columns)]

        interceptRandom = np.array(b['value'][(b['parameter'].str.contains('w0')) &
                                              (b['parameter'].str.contains(variable[1]))])

        betaRandom = np.array(b['value'][(b['parameter'].str.contains('B')) & (~b['parameter'].str.contains('B0'))])
        functionIntRandom = np.array(b['value'][b['parameter'].str.contains('B0')])

        singleFunction = (interceptRandom[0] + (wRandom * data.loc[:, data.columns != 'Pred']))
        toReturn = data.loc[:, data.columns != 'Pred']
        reLuFunction = pd.DataFrame(np.where(singleFunction > 0, toReturn, 0)).set_axis(
            data.loc[:, data.columns != 'Pred'].columns, axis=1)

        return (np.array(reLuFunction)).flatten().mean()

def optimizeNN (data, dependent, numberOfLayers, numberOfNodes, leaningRate, analytics = False):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import random

    layers = numberOfLayers
    nodes = numberOfNodes

    leaningRate = leaningRate

    pV = getParameterVector(data, layers, nodes).reset_index()
    del [pV['index']]

    trainingEpochs = 1
    max_iter = 20000

    minimizationPath = list()
    weights = pV
    MSEBefore = ((getPredictions(pV, data, dependent, layers, nodes) - data[dependent]) ** 2).mean()
    MSE = 0
    while (trainingEpochs < max_iter) & (MSEBefore > MSE):

        MSEBefore = (
                    (getPredictions(weights, data, dependent, layers, nodes) - data[dependent]) ** 2).mean()

        gradient = list()
        for param in range(len(weights['parameter'])):
            gr = computeGradient(weights['parameter'][param], data, layers, nodes)
            gradient.append(gr)
            moveGr = -(gr * leaningRate)
            w_old = weights['value']
            w_new = pd.concat([weights['parameter'], w_old + moveGr], axis=1).set_axis(['parameter', 'value'], axis=1)

        MSE = ((getPredictions(w_new, data, dependent, layers, nodes) - data[dependent]) ** 2).mean()

        # print(gradient)

        minimizationPath.append(MSE)

        weights = w_new

        leaningRate = leaningRate * 0.99

        print('Training Epochs:', trainingEpochs)
        print('Learning Rate', leaningRate)
        print('\n')
        print('MSE:', MSE)
        print('MSE before:', MSEBefore)

        trainingEpochs += 1

    predictionF = pd.concat([getPredictions(weights, data, 'Pred', layers, nodes),
                             getPredictions(pV, data, 'Pred', layers, nodes), data['Pred']],
                            axis=1).set_axis(['final Weights Prediction', 'Starting Weights', 'True Value'], axis=1)
    # print(predictionF)

    if analytics == True:

        minimizationPath = pd.Series(minimizationPath)
        plt.figure(figsize=(15, 5))
        plt.scatter(x=minimizationPath.index, y=minimizationPath)
        plt.title('Minimization Path (no Constraints)')
        plt.axhline(minimizationPath.min(), color='black', linestyle='dashed')

        plt.figure(figsize=(15, 5))
        plt.plot((predictionF['final Weights Prediction'] - predictionF['True Value']).cumsum(), color='blue')
        plt.plot((predictionF['Starting Weights'] - predictionF['True Value']).cumsum(), color='red')
        plt.title('Trend')
        plt.axhline(minimizationPath.min(), color='black', linestyle='dashed')

        plt.show()

    return weights