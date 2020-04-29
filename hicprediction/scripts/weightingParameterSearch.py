import click
import pandas as pd
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials
from hicprediction.training import checkTrainingset 
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt

@click.option('--datasetfile', '-d', type=click.Path(file_okay=True,dir_okay=False,exists=True,readable=True), help="trainingset from hicprediction createTrainingset.py")
@click.option('--outfile', '-o', type=click.Path(file_okay=True,dir_okay=False, writable=True), help="outfile for results")
@click.command()
def weightingParameterSearch(datasetfile, outfile):
    #load training set
    df, params, msg = checkTrainingset(datasetfile)
    if msg != None:
        msg = "Aborting.\n" + msg
        raise SystemExit(msg)
    dropList = ['first', 'second', 'chrom', 'avgRead', 'valid'] 
    df.drop(columns=dropList, inplace=True)

    #set the params for the random forest
    modelParamDict = dict(n_estimators=20, 
                            criterion='mse', 
                            max_depth=None, 
                            min_samples_split=2, 
                            min_samples_leaf=1, 
                            min_weight_fraction_leaf=0.0, 
                            max_features=1./3., 
                            max_leaf_nodes=None, 
                            min_impurity_decrease=0.0, 
                            min_impurity_split=None, 
                            bootstrap=True, 
                            oob_score=False, 
                            n_jobs=-1, 
                            random_state=5, 
                            verbose=0, 
                            ccp_alpha=0.0, 
                            max_samples=0.75)

    readMin = df['reads'].min() #normally zero
    readMax = df['reads'].max()
    readMean = df['reads'].mean()
    minReadFactor = readMin / readMean #normally zero
    maxReadFactor = readMax / readMean
    print("min: {:.3f}, max: {:.3f}, mean: {:.3f}".format(readMin, readMax, readMean))
    searchSpace = {
        'trainDf': hp.choice('trainDf', [df]),
        'estimatorParams': hp.choice('estimatorParams', [modelParamDict]),
        'bound1Float': hp.uniform('bound1Float', minReadFactor, maxReadFactor), #doesn't matter which one is smaller
        'bound2Float': hp.uniform('bound2Float', minReadFactor, maxReadFactor), #doesn't matter which one is smaller
        'factorFloat': hp.uniform('factorFloat', 0.1, 1000),
    }

    trials = Trials()
    best = fmin(fn=objectiveFunction, \
                algo=tpe.suggest, \
                space=searchSpace, \
                max_evals=2, \
                rstate=np.random.RandomState(35), \
                trials=trials)

    resDf = pd.DataFrame()
    for i, trialDict in enumerate(trials.trials):
        print("Trial " + str(i+1))
        for key in trialDict['result']:
            resDf.loc[i, str(key)] = trialDict['result'][key]
            print(" ", str(key), trialDict['result'][key])
        for key in trialDict['misc']['vals']:
            resDf.loc[i, str(key)] = trialDict['misc']['vals'][key]
            print(" ", str(key), trialDict['misc']['vals'][key][0])
    print("best values: ")
    for key in best:
        print(" ", str(key), best[key])
    
    resDf.to_csv(outfile)

def computeWeights(pDataframe, pBound1, pBound2, pFactorFloat):
    #equal weights for all samples as a start
    pDataframe['weights'] = 1 
    success = False
    #order boundaries and select
    if pBound1 < pBound2:
        lowerMask = pDataframe['reads'] >= pBound1
        upperMask = pDataframe['reads'] <= pBound2
    else:
        lowerMask = pDataframe['reads'] >= pBound2
        upperMask = pDataframe['reads'] <= pBound1
    numberOfWeightedSamples = pDataframe[lowerMask & upperMask].shape[0]
    numberOfUnweightedSamples = pDataframe.shape[0] - numberOfWeightedSamples
    if numberOfWeightedSamples == 0 or numberOfUnweightedSamples == 0:
        success = False
    else:
        print("number of non-emphasized samples: {:d}".format(numberOfUnweightedSamples))
        print("number of emphasized samples: {:d}".format(numberOfWeightedSamples))
        weightInt = int(np.round(pFactorFloat * numberOfUnweightedSamples / numberOfWeightedSamples))
        pDataframe.loc[lowerMask & upperMask, 'weights'] = weightInt
        weightSum = pDataframe.loc[lowerMask & upperMask, 'weights'].sum()
        print("weight given: {:d}".format(weightInt))
        print("weight sum non-emphasized samples: {:d}".format(numberOfUnweightedSamples))
        print("weight sum emphasized samples: {:d}".format(weightSum))
        print("target factor weighted/unweighted: {:.3f}".format(pFactorFloat))
        print("actual factor weighted/unweighted: {:.3f}".format(weightSum/numberOfUnweightedSamples))
        success = weightInt != 1
        
        #plot the weight vs. read count distribution after oversampling
        readMax = pDataframe['reads'].max()
        fig1, ax1 = plt.subplots()
        binSize = readMax / 100
        bins = pd.cut(pDataframe['reads'], bins=np.arange(0,readMax + binSize, binSize), labels=np.arange(0,readMax,binSize), include_lowest=True)
        printDf = pDataframe[['weights']].groupby(bins).sum()
        printDf.reset_index(inplace=True, drop=False)
        ax1.bar(printDf.reads, printDf.weights, width=binSize)
        ax1.set_xlabel("read counts")
        ax1.set_ylabel("weight sum")
        ax1.set_yscale('log')
        ax1.set_title("weight sum vs. read counts")
        figureTag = "/home/ralf/uniF/zeixx.png"        
        fig1.savefig(figureTag)
    return success

def objectiveFunction(paramDict):
    pEstimatorParams = paramDict['estimatorParams']
    pTrainDf = paramDict['trainDf']
    pBound1Float = paramDict['bound1Float']
    pBound2Float = paramDict['bound2Float']
    pFactorFloat = paramDict['factorFloat']
    status = STATUS_FAIL
    scoreMean = None
    attachmentDict = dict()
    #setup the estimator 
    model = RandomForestRegressor(**pEstimatorParams)
    #split training set into X, y
    dfX = pTrainDf.drop(columns=['reads'])
    dfY = pTrainDf[['reads']]
    #compute weights (stored directly in weight column of trainDf)
    readMean = pTrainDf['reads'].mean()
    bound1 = readMean * pBound1Float
    bound2 = readMean * pBound2Float
    success = computeWeights(pTrainDf, \
                            pBound1=bound1, \
                            pBound2=bound2, \
                            pFactorFloat=pFactorFloat)
    if success != True:
        status = STATUS_FAIL

    else:
        status = STATUS_OK
        dfWeights = pTrainDf[['weights']]
        kfoldCV = KFold(n_splits = 2, \
                        shuffle = True, \
                        random_state = 35)
        testScore = []
        for trainIndices, testIndices in kfoldCV.split(dfX):
            X_train = dfX.iloc[trainIndices,:]
            y_train = dfY.iloc[trainIndices,:].values.flatten()
            weights_train = dfWeights.iloc[trainIndices,:].values.flatten()
            model.fit(X_train, y_train, weights_train)
            # best possible test score is 1, so compute 1-score for minimizing
            # no weights used for test set (equal weight for all samples)
            X_test = dfX.iloc[testIndices,:]
            y_test = dfY.iloc[testIndices,:].values.flatten()
            testScore.append(1 - model.score(X_test, y_test)) 
        scoreMean = np.mean(testScore)
        scoreStd = np.std(testScore)
        attachmentDict = {'stdDev': scoreStd}
    returnDict = {'status': status, 'loss': scoreMean, 'attachments': attachmentDict}
    return returnDict




if __name__ == "__main__":
    weightingParameterSearch()