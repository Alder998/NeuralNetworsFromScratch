# Questa libreria serve per dare un design OOP ai metodi per creare una rete neurale
# Questo serve anche per creare nuovi Spinoff di reti, come CNN, o, molto più probabilmente, RNN.

# Allo stesso tempo questo serve anche per Fare in modo che la rete neurale approcci più tipi di
# problemi (regressione, classificazione, classificazione multipla).

# Classe inizializzazione

class Regression:
    name = "NN From Scratch"

    def __init__(self, inputLayer, shape):
        self.shape = shape
        self.inputLayer = inputLayer
        pass

    # Metodo per ottenere la struttura della Rete Neurale per un problema di regressione

    def structure(self):

        import pandas as pd
        import numpy as np
        import random

        numberOfNodes = self.shape
        numberOfLayers = len(numberOfNodes)
        # numberOfNodes = numberOfNodes

        # Costruiamo i pesi

        weights = list()
        beta = list()
        interceptsW = list()
        neuralIntercept = ['B0']
        for layer in range(1, numberOfLayers + 1):

            if layer == 1:
                nodes = self.inputLayer - 1  # considera la colonna che devi prevedere
            if layer != 1:
                nodes = numberOfNodes[layer - 2]

            # print(nodes)

            # Inizializza ogni layer con la relativa lunghezza
            # Definiamo le operazioni da svolgere nel singolo nodo

            for node in range(1, numberOfNodes[layer - 1] + 1):

                # print(node)

                b = 'B' + str(node)
                w0 = 'w0' + str(node) + '_' + str(layer)

                beta.append(b)
                interceptsW.append(w0)

                for parameter in range(1, nodes + 1):
                    w = 'W' + str(node) + '.' + str(parameter) + '_' + str(layer)

                    # print('Layer', layer, ', Node', node, ', Parameter', parameter)

                    weights.append(w)

                    # All'interno di ogni singolo nodo, infatti, c'è l'equazione di una regressione lineare, che
                    # Definiamo come Ak

        lastLayerLength = numberOfNodes[len(numberOfNodes) - 1]
        beta = beta[len(beta) - lastLayerLength: len(beta)]

        wRandom = pd.concat([pd.Series(weights), pd.Series(np.abs(np.random.normal(loc = 0.3, scale = 0.01, size=len(weights))))],
                            axis=1).set_axis(['parameter', 'value'], axis=1)
        interceptRandom = pd.concat([pd.Series(interceptsW), pd.Series(np.abs(np.random.normal(loc = 0.3, scale = 0.01,
                                                                                               size=len(interceptsW))))],
                                    axis=1).set_axis(['parameter', 'value'], axis=1)
        betaRandom = pd.concat([pd.Series(beta), pd.Series(np.abs(np.random.normal(loc = 0.3, scale = 0.01, size=len(beta))))],
                               axis=1).set_axis(['parameter', 'value'], axis=1)
        functionIntRandom = pd.concat([pd.Series(neuralIntercept), pd.Series(random.random() * random.choice([-1, 1]))],
                                      axis=1).set_axis(['parameter', 'value'], axis=1)

        # L'output va interpretato così:
        # res[0] = Beta che vanno inseriti nella f(X) finale (numero nodi)
        # res[1] = Intercette per ogni nodo (numero nodi)
        # res[2] = pesi per ogni nodo (numero nodi x numero predictors)
        # res[3] = intercetta della rete (un solo valore)

        return pd.concat([betaRandom, interceptRandom, wRandom, functionIntRandom], axis=0)


    def getPredictions(self, parameterVector, data, dependent):

        import pandas as pd
        import numpy as np

        b = parameterVector

        wRandom = np.array(b['value'][b['parameter'].str.contains('W')])
        interceptRandom = np.array(b['value'][b['parameter'].str.contains('w0')])
        betaRandom = np.array(b['value'][(b['parameter'].str.contains('B')) & (~b['parameter'].str.contains('B0'))])
        functionIntRandom = np.array(b['value'][b['parameter'].str.contains('B0')])

        # Costruiamo ogni singolo Nodo usando i dati random e una funzione non lineare reLu

        functionListNode = list()
        nodeO = 0
        while nodeO < self.numberOfNodes:
            regrWeights = (wRandom[(len(data.columns)-1)*nodeO : ((len(data.columns)-1)*nodeO + (len(data.columns)-1))])
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


    def getAllLayerValues(self, parameterVector, data, dependent, return_nodes=False):

        numberOfNodes = self.shape
        numberOfLayers = len(numberOfNodes)

        import numpy as np
        import pandas as pd

        # data = data.loc[:,data.columns != dependent]
        # data = data.set_axis([np.arange(1, len(data.columns) + 1)], axis = 1)

        b = parameterVector
        b = b.reset_index()
        del [b['index']]

        wRandomSt = (b[b['parameter'].str.contains('W')])
        interceptRandomSt = (b[b['parameter'].str.contains('w0')])

        # Costruiamo ogni singolo Nodo usando i dati random e una funzione non lineare reLu

        lastLayer = list()
        AllLayers = list()

        functionListNode = list()
        target = data.loc[:, data.columns != dependent]
        layerO = 1
        while layerO < numberOfLayers + 1:

            wRandom = np.array(wRandomSt['value'][wRandomSt['parameter'].str.contains('_' + str(layerO))])

            # print(wRandomSt[wRandomSt['parameter'].str.contains('_' + str(layerO))])

            interceptRandom = np.array(
                interceptRandomSt['value'][interceptRandomSt['parameter'].str.contains('_' + str(layerO))])

            nodeO = 0
            while nodeO < numberOfNodes[layerO - 1]:

                # print(len(interceptRandom[nodeO]))

                # print(interceptRandom[2])

                if layerO == 1:
                    regrWeights = (wRandom[nodeO * (len(data.columns) - 1): nodeO * (len(data.columns) - 1) +
                                                                            (len(data.columns) - 1)])

                if layerO != 1:
                    regrWeights = (wRandom[nodeO * numberOfNodes[layerO - 2]: nodeO * numberOfNodes[layerO - 2] +
                                                                              numberOfNodes[layerO - 2]])

                # print(regrWeights)
                # print(target)

                # singleFunction è l'operazione che viene fatta in ogni nodo. Bisogna fare in modo che un DF con
                # (numero di input) colonne diventi un DF con (nodi al primo layer) colonne. Per questo dobbiamo
                # sommare l'intercetta, come facciamo adesso, ma successivamente dobbiamo fare la somma (con trasponi,
                # somma, trasponi) e poi concatenare i risultati.

                singleFunction = (interceptRandom[nodeO] + (regrWeights * target))
                reLuFunction = pd.DataFrame(np.where(singleFunction > 0, singleFunction, 0))

                AllLayers.append(reLuFunction)

                reLuFunction = reLuFunction.transpose().sum().transpose()

                functionListNode.append(reLuFunction)

                nodeO += 1

            functionListNode1 = pd.concat([series for series in functionListNode], axis=1)
            target = functionListNode1

            lastLayer.append(target)

            functionListNode = list()

            layerO += 1

        # AllLayers deve avere le colonne chiamate nel modo giusto, e cioè shiftato di uno

        if return_nodes == False:

            results = list()
            for series in AllLayers:
                j = series.set_axis([np.arange(1, len(series.columns) + 1)], axis=1)
                results.append(j)

            return results

        if return_nodes == True:

            results = list()
            for series in lastLayer:
                j = series.set_axis([np.arange(1, len(series.columns) + 1)], axis=1)
                results.append(j)

            return results


    def computeGradient(self, variable, data, dependent):

        import numpy as np
        import pandas as pd

        numberOfNodes = self.shape
        numberOfLayers = len(numberOfNodes)

        b = self.structure()

        b.to_excel(r"C:\Users\39328\OneDrive\Desktop\Davide\Velleità\NN.xlsx")

        if variable == 'B0':
            # Single-Layer o Multi-Layer questo risultato non cambia

            return 1

        if (variable[0] == 'B') & (variable != 'B0'):
            # Il gradiente in base Bk (== Beta nel nostro modello) risulta Ak (dove Ak è il nodo associato al beta
            # di cui si vuole cacolare il gradiente). Se il parametro è B1, allora il gradiente sarà A1. Gli A che vanno presi
            # sono SOLO quelli all'ultimo Layer (== Tutti i nodi dell'ultimo Layer). Per fare ciò, ci serve il valore
            # della prediction all'ultimo layer. Dopo averla ottenuta, dobbiamo sapere distinguere per il valore che
            # ci interessa.

            # Quindi prendiamo l'ultimo layer

            allLayers = self.getAllLayerValues(b, data, dependent, return_nodes=True)
            lastLayer = allLayers[len(allLayers) - 1]

            # Adesso ci serve prendere la colonna dell'ultimo layer corrispondente al nodo della nostra variabile beta

            node = int(variable[1:len(variable)])

            result = lastLayer[node]

            return (np.array(result)).flatten().mean()

        if (variable[0] == 'w'):
            node = variable[1: variable.find('_')]
            layer = variable[variable.find('_') + 1: len(variable)]
            prLayer = int(layer) - 1

            # Prendiamo tutti i layer
            allLayers = self.getAllLayerValues(b, data, dependent, return_nodes=True)

            # scegliamo il layer precedente a quello attuale
            PrLayer = allLayers[prLayer]

            # Prendiamo la variabile k (== il nodo che prendiamo dalla variabile)
            PrLayer = PrLayer[int(node)]  # La size è (size, 1)

            result = np.where(PrLayer > 0, 1, 0)  # La funzione è una reLu

            return (np.array(result)).flatten().mean()

        if (variable[0] == 'W'):

            # Calcolare il gradiente della funzione in abse ad ognuno dei pesi è la parte più complicata. Infatti, se si tratta
            # del primo layer, allora il gradiente sarà semplicemente g(Xk) (X = Input). per i layer dopo il primo, invece, il
            # risultato del gradiente sarà g(A(i-1)k), dove A(i-1) è il layer precedente a quello esaminato. g è la funzione di
            # attivazione che scegliamo (nel nostro caso è una reLu).

            # Definiamo alcuni numeri utili legati alla variabile che scegliamo: il numero del layer precedente, il numero del
            # Layer attuale, il numero di nodo e il numero di parametro

            layerNumber = int(variable[variable.find('_') + 1 : len(variable)])

            connectionPerNode = len(data.loc[:, data.columns != dependent].columns)

            parameter = variable[variable.find('.') + 1:variable.find('_')]
            layer = variable[variable.find('_') + 1: len(variable)]
            node = variable[1: variable.find('.')]

            prLayer = int(layer) - 1

            # Prendiamo tutti i layer
            allLayers = self.getAllLayerValues(b, data, dependent)
            allNodes = self.getAllLayerValues(b, data, dependent, return_nodes=True)

            if layerNumber == 1:
                # scegliamo il primo layer (quello con le X)
                firstLayer = allLayers[0]

                # Prendiamo la variabile k (== il nodo che prendiamo dalla variabile)
                firstLayer = firstLayer[int(parameter)]  # La size è (size, 1)

                result = np.where(firstLayer > 0, firstLayer, 0)  # La funzione è una reLu

                return (np.array(result)).flatten().mean()

            if layerNumber > 1:
                # scegliamo il layer precedente a quello attuale
                PrLayer = allNodes[prLayer]

                # Prendiamo la variabile k (== il nodo che prendiamo dalla variabile)

                PrLayer = PrLayer[(int(node))]  # La size è (size, 1)

                result = np.where(PrLayer > 0, PrLayer, 0)  # La funzione è una reLu

                return (np.array(result)).flatten().mean()


    def fit(self, data, dependent, leaningRate, decreasingRate=0.99, analytics=False):

        import pandas as pd
        import matplotlib.pyplot as plt

        leaningRate = leaningRate

        pV = self.structure().reset_index()
        del [pV['index']]

        trainingEpochs = 1
        max_iter = 20000

        lRControl = leaningRate

        minimizationPath = list()
        weights = pV
        MSEBefore = ((self.getPredictions2(pV, data, dependent) - data[dependent]) ** 2).mean()
        MSE = 0
        while (trainingEpochs < max_iter) & (MSEBefore > MSE):

            MSEBefore = (
                    (self.getPredictions2(weights, data, dependent) - data[dependent]) ** 2).mean()

            gradient = list()
            for param in range(len(weights['parameter'])):
                gr = self.computeGradient(weights['parameter'][param], data, dependent)
                gradient.append(gr)
                moveGr = -(gr * leaningRate)
                w_old = weights['value']
                w_new = pd.concat([weights['parameter'], w_old + moveGr], axis=1).set_axis(['parameter', 'value'],
                                                                                           axis=1)

            MSE = ((self.getPredictions2(w_new, data, dependent) - data[dependent]) ** 2).mean()

            # print(gradient)

            minimizationPath.append(MSE)

            weights = w_new

            leaningRate = max((leaningRate * decreasingRate), lRControl*0.05)

            print('Training Epochs:', trainingEpochs)
            print('Learning Rate', leaningRate)
            print('\n')
            print('MSE:', MSE)
            print('MSE before:', MSEBefore)

            trainingEpochs += 1

        predictionF = pd.concat([self.getPredictions2(weights, data, 'Pred'),
                                 self.getPredictions2(pV, data, 'Pred'), data['Pred']],
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


    def getPredictions2(self, parameterVector, data, dependent):

        import numpy as np
        import pandas as pd

        numberOfNodes = self.shape
        numberOfLayers = len(numberOfNodes)

        # data = data.loc[:,data.columns != dependent]
        # data = data.set_axis([np.arange(1, len(data.columns) + 1)], axis = 1)

        b = parameterVector
        b = b.reset_index()
        del [b['index']]

        wRandomSt = (b[b['parameter'].str.contains('W')])
        interceptRandomSt = (b[b['parameter'].str.contains('w0')])
        betaRandom = np.array(b['value'][(b['parameter'].str.contains('B')) & (~b['parameter'].str.contains('B0'))])

        functionIntRandom = np.array(b['value'][b['parameter'].str.contains('B0')])

        # Costruiamo ogni singolo Nodo usando i dati random e una funzione non lineare reLu

        lastLayer = list()

        functionListNode = list()
        target = data.loc[:, data.columns != dependent]
        layerO = 1
        while layerO < numberOfLayers + 1:

            wRandom = np.array(wRandomSt['value'][wRandomSt['parameter'].str.contains('_' + str(layerO))])

            # print(wRandomSt[wRandomSt['parameter'].str.contains('_' + str(layerO))])

            interceptRandom = np.array(interceptRandomSt['value'][interceptRandomSt['parameter'].str.contains('_' +
                                                                                                              str(layerO))])

            nodeO = 0
            while nodeO < numberOfNodes[layerO - 1]:

                # print(len(interceptRandom[nodeO]))

                # print(interceptRandom[2])

                if layerO == 1:
                    regrWeights = (wRandom[nodeO * (len(data.columns) - 1): nodeO * (len(data.columns) - 1) +
                                                                            (len(data.columns) - 1)])

                if layerO != 1:
                    regrWeights = (wRandom[nodeO * numberOfNodes[layerO - 2]: nodeO * numberOfNodes[layerO - 2] +
                                                                              numberOfNodes[layerO - 2]])

                # print(regrWeights)
                # print(target)

                # singleFunction è l'operazione che viene fatta in ogni nodo. Bisogna fare in modo che un DF con
                # (numero di input) colonne diventi un DF con (nodi al primo layer) colonne. Per questo dobbiamo
                # sommare l'intercetta, come facciamo adesso, ma successivamente dobbiamo fare la somma (con trasponi,
                # somma, trasponi) e poi concatenare i risultati.

                singleFunction = (interceptRandom[nodeO] + (regrWeights * target))
                reLuFunction = pd.DataFrame(np.where(singleFunction > 0, singleFunction, 0))

                lastLayer.append(target)

                reLuFunction = reLuFunction.transpose().sum().transpose()

                functionListNode.append(reLuFunction)

                nodeO += 1

            functionListNode1 = pd.concat([series for series in functionListNode], axis=1)
            target = functionListNode1

            functionListNode = list()

            layerO += 1

        lastLayer = lastLayer[len(lastLayer) - numberOfNodes[len(numberOfNodes) - 1]: len(lastLayer)]

        # Per come abbiamo costruit il tutto, il layer finale ce lo dobbiamo costruire da soli. Infatti, il nostro layer finale
        # ha forma (connessioni nel layer precedente X numero nodi nel layer precedente). Quindi, se nel layer precedente
        # avevamo 8 nodi, e l'ultimo layer ha 3 nodi, avremo 3 serie da 8 colonne ciascuna. Dobbiamo puntare ad avere invece
        # una serie singola, ma con tre colonna (una per layer).

        lastLastLayer = list()
        for df in lastLayer:
            dfU = df.transpose().sum().transpose()  # somma per riga
            lastLastLayer.append(dfU)

        lastLastLayer = pd.concat([series for series in lastLastLayer], axis=1)

        # Abbiamo costruito ogni singolo nodo, ora dobbiamo inserirlo nell'equazione generale

        predList = list()
        for SingleNodeValues in range(0, len(lastLastLayer.columns) - 1):
            predList.append(betaRandom[SingleNodeValues] * lastLastLayer[SingleNodeValues])

        equationResult = list()
        for predictorEquationsResults in predList:
            equationResult.append((functionIntRandom[0] + predictorEquationsResults))

        equationResult = pd.concat([series for series in equationResult], axis=1)
        equationResult = equationResult.transpose().sum().transpose()

        return equationResult


class Classification:
    name = "NN From Scratch - Classification"

    def __init__(self, inputLayer, shape, classes):
        self.shape = shape
        self.inputLayer = inputLayer
        self.classes = classes
        pass

    # Metodo per ottenere la struttura della Rete Neurale per un problema di regressione

    def structure (self):

        import pandas as pd
        import numpy as np
        import random

        numberOfNodes = self.shape
        numberOfLayers = len(numberOfNodes)

        # Costruiamo i pesi

        weights = list()
        beta = list()
        interceptsW = list()
        neuralIntercept = ['B0']
        for layer in range(1, numberOfLayers + 1):

            # Inizializza ogni layer con la relativa lunghezza
            # Definiamo le operazioni da svolgere nel singolo nodo

            for node in range(1, numberOfNodes[layer - 1] + 1):

                b = 'B' + str(node)
                w0 = 'w0' + str(node) + '_' + str(layer)

                beta.append(b)
                interceptsW.append(w0)

                for parameter in range(1, self.inputLayer):
                    w = 'W' + str(node) + str(parameter) + '_' + str(layer)

                    weights.append(w)

                    # All'interno di ogni singolo nodo, infatti, c'è l'equazione di una regressione lineare, che
                    # Definiamo come Ak

        # Ora bisogna creare artificialmente i nodi responsabili della classificazione, che è indipendente dal layer

        finalClassificationNodes = list()
        for node in range(0, numberOfNodes[len(numberOfNodes) - 1]):

            for classNode in range(0, self.classes):
                finalClassificationNodes.append('Class' + str(node) + str(classNode))

        # Ora calcoliamo l'intercetta per la classificazione

        classIntercept = list()
        for classNumber in range(0, self.classes):
            classIntercept.append('Class0' + str(classNumber))

        wRandom = pd.concat([pd.Series(weights), pd.Series(np.random.uniform(size=len(weights)))],
                            axis=1).set_axis(['parameter', 'value'], axis=1)

        interceptRandom = pd.concat([pd.Series(interceptsW), pd.Series(np.random.uniform(size=len(interceptsW)))],
                                    axis=1).set_axis(['parameter', 'value'], axis=1)

        finalClassificationNodes = pd.concat([pd.Series(finalClassificationNodes),
                                              pd.Series(np.random.uniform(size=len(finalClassificationNodes)))],
                                             axis=1).set_axis(['parameter', 'value'], axis=1)

        classIntercept = pd.concat([pd.Series(classIntercept), pd.Series(np.random.uniform(size=len(classIntercept)))],
                                   axis=1).set_axis(['parameter', 'value'], axis=1)


        return pd.concat([wRandom, interceptRandom, classIntercept, finalClassificationNodes],
                         axis=0)