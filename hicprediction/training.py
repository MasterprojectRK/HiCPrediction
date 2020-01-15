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
    ### check if the dataset contains reads, otherwise it cannot be used for training
    nanreadMask = df['reads'] == np.nan
    if not df[nanreadMask].empty:
        #any nans coming from bad bins in the HiC-matrix must be replaced when creating the dataset
        #so there should not be any nans left
        msg = "dataset {0:s} cannot be used for training" 
        msg += "because it contains no target reads"
        raise ValueError(msg.format(traindatasetfile))
    
    ### store the training parameters for later reference and file name creation
    params['conversion'] = conversion
    params['noDistance'] = noDist
    params['noMiddle'] = noMiddle
    params['noStartEnd'] = noStartEnd
        
    ### create model with desired parameters
    model = sklearn.ensemble.RandomForestRegressor(max_features=pMaxFeat, random_state=5,\
                    n_estimators=pNrOfTrees, n_jobs=4, verbose=2, criterion='mse')
    
    ### replace infinity and nan values. There shouldn't be any.
    rowsBefore = df.shape[0]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(value=0, inplace=True)

    ### drop columns that should not be used for training
    dropList = ['first', 'second', 'chrom', 'reads', 'avgRead']
    if noDist:
        dropList.append('distance')    
    if noMiddle:
        if params['method'] == 'oneHot':
            dropList.append('middleProt')
        elif params['method'] == 'multiColumn':
            numberOfProteins = int((df.shape[1] - 6) / 3)
            for protein in range(numberOfProteins):
                dropList.append(str(protein + numberOfProteins))
        else:
            raise NotImplementedError()
    if noStartEnd:
        if params['method'] == 'oneHot':
            dropList.append('startProt')
            dropList.append('endProt')
        elif params['method'] == 'multiColumn':
            numberOfProteins = int((df.shape[1] - 6) / 3)
            for protein in range(numberOfProteins):
                dropList.append(str(protein))
                dropList.append(str(protein + 2 * numberOfProteins))
        else:
            raise NotImplementedError()
    X = df[df.columns.difference(dropList)]
        
    ### apply conversion
    if conversion == 'none':
        y = df['reads']
    elif conversion == 'standardLog':
        y = np.log(df['reads']+1)

    ## train model and store it
    model.fit(X, y)
    modelTag = createModelTag(params) + ".z"
    modelFileName = os.path.join(modeloutputdirectory, modelTag)
    joblib.dump((model, params), modelFileName, compress=True ) 
    print("\n")

if __name__ == '__main__':
    train()
