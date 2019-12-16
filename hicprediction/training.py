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
def train(modeloutputdirectory, conversion, traindatasetfile, nodist, nomiddle):
    """
    Wrapper function for click
    """
    training(modeloutputdirectory, conversion, traindatasetfile, nodist, nomiddle)

def training(modeloutputdirectory, conversion, traindatasetfile, noDist, noMiddle):
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
    modelTag = createModelTag(params) + ".z"
    modelFileName = os.path.join(modeloutputdirectory, modelTag)
    #exists = os.path.isfile(modelFileName)
    exists = False
    if not exists:
        ### create model with desired parameters
        model = sklearn.ensemble.RandomForestRegressor(max_features='sqrt',random_state=5,\
                    n_estimators =10,n_jobs=4, verbose=2, criterion='mse')
        df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(value=0)
        ### drop columns that should not be used for training
        dropList = ['first', 'second', 'chrom', 'reads', 'avgRead']
        if noDist:
            dropList.append('distance')
        if noMiddle:
            numberOfProteins = int((df.shape[1] - 6) / 3)
            for protein in range(numberOfProteins):
                dropList.append(str(protein + numberOfProteins))
        X = df[df.columns.difference(dropList)]
        ### apply conversion
        if conversion == 'none':
            y = df['reads']
        elif conversion == 'standardLog':
            y = np.log(df['reads']+1)

        ## train model and store it
        model.fit(X, y)
        params['noDistance'] = noDist
        params['noMiddle'] = noMiddle
        joblib.dump((model, params), modelFileName,compress=True ) 
        print("\n")
    else:

        print("Skipped creating model that already existed: " + modelFileName)

if __name__ == '__main__':
    train()
