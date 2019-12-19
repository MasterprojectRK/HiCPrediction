#!/usr/bin/env python3
import os
os.environ['NUMEXPR_MAX_THREADS'] = '16'
os.environ['NUMEXPR_NUM_THREADS'] = '8'
import hicprediction.configurations as conf
import click
import joblib
import sklearn.ensemble
import numpy as np
from hicprediction.tagCreator import createModelTag
import sys

""" Module responsible for the training of the regressor with data sets.
"""
@conf.train_options
@click.command()
def train(modeloutputdirectory, conversion, trees, maxfeat, traindatasetfile, nodist, nomiddle, nostartend):
    """
    Wrapper function for click
    """
    if maxfeat == 'none':
        maxfeat = None
    training(modeloutputdirectory, conversion, trees, maxfeat, traindatasetfile, nodist, nomiddle, nostartend)

def training(modeloutputdirectory, conversion, pNrOfTrees, pMaxFeat, traindatasetfile, noDist, noMiddle, noStartEnd):
    """
    Train function
    Attributes:
        modeloutputdirectory -- path to desired store location of models
        conversion --  read values conversion method 
        traindatasetfile -- input data set for training
    """
    ### checking extensions of files
    if not conf.checkExtension(traindatasetfile, ".z"):
        msg = "Aborted. Data set {0:s} must have .z file extension"
        sys.exit(msg.format(traindatasetfile))
    ### load data set and set parameters
    df, params = joblib.load(traindatasetfile)
    params['conversion'] = conversion
    params['noDistance'] = noDist
    params['noMiddle'] = noMiddle
    params['noStartEnd'] = noStartEnd
    modelTag = createModelTag(params) + ".z"
    modelFileName = os.path.join(modeloutputdirectory, modelTag)
    #exists = os.path.isfile(modelFileName)
    exists = False
    if not exists:
        ### create model with desired parameters
        model = sklearn.ensemble.RandomForestRegressor(max_features=pMaxFeat, random_state=5,\
                    n_estimators=pNrOfTrees, n_jobs=4, verbose=2, criterion='mse')
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(value=0, inplace=True)
        ### drop columns that should not be used for training
        dropList = ['first', 'second', 'chrom', 'reads', 'avgRead']
        if noDist:
            dropList.append('distance')
        if noMiddle:
            dropList.append('middleProt')
        if noStartEnd:
            dropList.append('startProt')
            dropList.append('endProt')    
        X = df[df.columns.difference(dropList)]
        
        ### apply conversion
        if conversion == 'none':
            y = df['reads']
        elif conversion == 'standardLog':
            y = np.log(df['reads']+1)

        ## train model and store it
        model.fit(X, y)
        joblib.dump((model, params), modelFileName, compress=True ) 
        print("\n")
    else:

        print("Skipped creating model that already existed: " + modelFileName)

if __name__ == '__main__':
    train()
