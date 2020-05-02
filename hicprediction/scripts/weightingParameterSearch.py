import click
import pandas as pd
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials
from hicprediction.training import checkTrainingset, computeWeights 
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
import numpy as np
#import matplotlib.pyplot as plt
import joblib

@click.option('--datasetfile', '-d', type=click.Path(file_okay=True,dir_okay=False,exists=True,readable=True), help="trainingset from hicprediction createTrainingset.py")
@click.option('--outfile', '-o', type=click.Path(file_okay=True,dir_okay=False, writable=True), help="outfile for results")
@click.option('--weightfactor', '-w', type=click.FloatRange(min=1.0), default=100.0, help="max search range for (weight of emphasized samples / weight of non-emph. samples)")
@click.option('--maxevals', '-me', type=click.IntRange(min=10), default=200, help="max. nr. of of search runs")
@click.option('--weightingType', '-wt', type=click.Choice(choices=['reads', 'proteinFeatures']), default='reads', help="compute weights based on reads (default) or protein feature values")
@click.option('--featList', '-fl', multiple=True, type=str, default=['0','12','24'], required=True, help="name of features according to which the weight is computed; default is 0, 12, 24; only relevant if wt=proteinFeatures")
@click.command()
def weightingParameterSearch(datasetfile, outfile, weightfactor, maxevals, weightingtype,featlist):
    ### use Cross-Validation and tree-structured Parzen estimators to find a range
    ### and factor for weighting samples in a hicprediction dataset
    ### the dataset must be suitable for training, i.e. it must contain a 'reads' column
    ### The weighting can be according to interaction count ('reads') or protein Feature values ('proteinFeatures')
    
    #check inputs first
    if weightingtype == 'proteinFeatures' and not featlist or len(featlist) == 0:
        msg = "weightingType proteinFeatures only possible when featList not empty"
        raise SystemExit(msg)
    elif weightingtype == 'proteinFeatures':
        columnList = featlist
    else:
        columnList = ['reads']
    #load training set
    df, params, msg = checkTrainingset(datasetfile)
    if msg != None:
        msg = "Aborting.\n" + msg
        raise SystemExit(msg)
    for element in columnList:
        if element not in df.columns:
            msg = "feature {:s} not contained in training set".format(str(element))
            raise SystemExit(msg)
    #kick columns from dataset which are not required for cross-validation
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

    minValue = df[columnList[0]].min()
    maxValue = df[columnList[0]].max()
    meanValue = df[columnList[0]].mean()
    minReadFactor = minValue / meanValue
    maxReadFactor = maxValue / meanValue
    print("min: {:.3f}, max: {:.3f}, mean: {:.3f}".format(minValue, maxValue, meanValue))
    searchSpace = {
        'trainDf': hp.choice('trainDf', [df]),
        'estimatorParams': hp.choice('estimatorParams', [modelParamDict]),
        'bound1Float': hp.uniform('bound1Float', minReadFactor, maxReadFactor), #doesn't matter which one is smaller
        'bound2Float': hp.uniform('bound2Float', minReadFactor, maxReadFactor), #doesn't matter which one is smaller
        'factorFloat': hp.uniform('factorFloat', 0.1, weightfactor),
        'params': hp.choice('params', [params]),
        'columnList': hp.choice('columnList', [columnList]),
    }

    trials = Trials()
    best = fmin(fn=objectiveFunction, \
                algo=tpe.suggest, \
                space=searchSpace, \
                max_evals=maxevals, \
                rstate=np.random.RandomState(35), \
                trials=trials)

    resDf = pd.DataFrame()
    for i, trialDict in enumerate(trials.trials):
        print("Trial " + str(i+1))
        for key in trialDict['result']:
            resDf.loc[i, str(key)] = trialDict['result'][key]
            print(" ", str(key), trialDict['result'][key])
        for key in trialDict['misc']['vals']:
            resDf.loc[i, str(key)] = trialDict['misc']['vals'][key][0]
            print(" {:s}: {:.3f}".format(str(key), trialDict['misc']['vals'][key][0]))
    print("best values: ")
    for key in best:
        print(" {:s} {:.3f}".format(str(key), best[key]))
    resDf.to_csv(outfile)
    joblib.dump(value=trials, filename=outfile + ".z", compress=3)


def objectiveFunction(paramDict):
    pEstimatorParams = paramDict['estimatorParams']
    pTrainDf = paramDict['trainDf']
    pBound1Float = paramDict['bound1Float']
    pBound2Float = paramDict['bound2Float']
    pFactorFloat = paramDict['factorFloat']
    pParams = paramDict['params']
    pColumnList = paramDict['columnList']
    status = STATUS_FAIL
    scoreMean = None
    attachmentDict = {
        'stdDev': None,
    }
    #setup the estimator 
    model = RandomForestRegressor(**pEstimatorParams)
    #split training set into X, y
    dfX = pTrainDf.drop(columns=['reads'])
    dfY = pTrainDf[['reads']]
    #compute weights (stored directly in weight column of trainDf)
    meanValue = pTrainDf[pColumnList[0]].mean()
    bound1 = meanValue * pBound1Float
    bound2 = meanValue * pBound2Float
    success = computeWeights(pTrainDf, \
                            pBound1=bound1, \
                            pBound2=bound2, \
                            pFactorFloat=pFactorFloat, \
                            pParams=pParams,\
                            pColumnsToUse=pColumnList)

    if success != True:
        status = STATUS_FAIL

    else:
        status = STATUS_OK
        dfWeights = pTrainDf[['weights']]
        kfoldCV = KFold(n_splits = 5, \
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
        attachmentDict['stdDev'] = scoreStd
    returnDict = {'status': status, 'loss': scoreMean, 'attachments': attachmentDict}
    return returnDict

if __name__ == "__main__":
    weightingParameterSearch() # pylint: disable=no-value-for-parameter