import click
import pandas as pd
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL
from hicprediction.createTrainingSet import checkTrainingset
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
import numpy as np

@click.option('--traindatafile', '-tdf', type=click.Path(file_okay=True,dir_okay=False,exists=True,readable=True), help="trainingset from hicprediction createTrainingset.py")
@click.command()
def weightingParameterSearch(traindatafile):
    #load training set
    df, params, msg = checkTrainingset(traindatafile)
    if msg != None:
        msg = "Aborting.\n" + msg
        raise SystemExit(msg)

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
                            verbose=2, 
                            ccp_alpha=0.0, 
                            max_samples=0.75)
    model = RandomForestRegressor(**modelParamDict)
    

def computeWeights(pDataframe, pLowerBoundFloat, pUpperBoundFloat, pFactorFloat):
    return [0]




def objectiveFunction(pTrainDf, pEstimator, pLowerBoundFloat, pUpperBoundFloat, pFactorFloat):
    
    #prepare the training set, drop unnecessary columns
    dropList = ['first', 'second', 'chrom', 'reads', 'avgRead', 'valid', 'weights']
    dfX = pTrainDf.drop(columns=dropList)
    dfY = pTrainDf[['reads']]
    #compute weights (stored directly in weight column of trainDf)
    computeWeights(pTrainDf, \
                            pLowerBoundFloat=pLowerBoundFloat, \
                            pUpperBoundFloat=pUpperBoundFloat, \
                            pFactorFloat=pFactorFloat)
    dfWeights = pTrainDf[['weights']]

    kfoldCV = KFold(n_splits = 5, \
                        shuffle = True, \
                        random_state = 35)
    testScore = []
    for trainIndices, testIndices in kfoldCV.split(dfX):
            X_train = dfX.iloc[trainIndices,:]
            y_train = dfY.iloc[trainIndices,:]
            weights_train = dfWeights.iloc[trainIndices,:]
            pEstimator.fit(X_train, y_train, weights_train)
            # best possible test score is 1, so compute 1-score
            # for minimizing
            # no weights for test set
            X_test = dfX.iloc[testIndices,:]
            y_test = dfY.iloc[testIndices,:]
            testScore.append(1 - pEstimator.score(X_test, y_test)) 
    scoreMean = np.mean(testScore)
    #scoreStd = np.std(testScore)
    returnDict = {'status': STATUS_OK, 'loss': scoreMean}
    return returnDict


if __name__ == "__main__":
    weightingParameterSearch()