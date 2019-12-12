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
def train(modeloutputdirectory, conversion, traindatasetfile):
    """
    Wrapper function for click
    """
    training(modeloutputdirectory, conversion, traindatasetfile)

def training(modeloutputdirectory, conversion, traindatasetfile):
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
        model = sklearn.ensemble.RandomForestRegressor(max_features=None,random_state=5,\
                    n_estimators =30,n_jobs=4, verbose=2, criterion='mse')
        df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(value=0)
        print(df.shape, 'shape before')
        df.drop_duplicates(keep='first', inplace=True, subset=['0', '1', '2','distance','reads','avgRead'])
        print(df.shape, 'shape after')

        ### eliminate columns that should not be used for training
        X = df[df.columns.difference(['first', 'second','chrom', 'reads', 'avgRead'])]
        #dist0 = X['distance'] == 0
        # print(X[dist0])
        # print('before dropping', df.shape)
        # X.drop_duplicates(keep='first',inplace=True)
        # print('after dropping', df.shape)
        ### apply conversion
        if conversion == 'none':
            y = df['reads']
        elif conversion == 'standardLog':
            y = np.log(df['reads']+1)

        ## train model and store it
        model.fit(X, y)
        joblib.dump((model, params), modelFileName,compress=True ) 
        print("\n")
    else:

        print("Skipped creating model that already existed: " + modelFileName)

if __name__ == '__main__':
    train()
