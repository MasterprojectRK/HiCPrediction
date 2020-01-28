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
import matplotlib.pyplot as plt
from scipy.stats import expon
import pandas as pd

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
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(value=0, inplace=True)


    ### oversampling
    #plot the raw read distribution first
    print("oversampling")
    fig1, ax1 = plt.subplots(figsize=(8,6))
    df.hist(column='reads', bins=100, ax=ax1, density=True) 
    ax1.set_xlabel("read counts")
    ax1.set_ylabel("normalized frequency")
    ax1.set_title("raw read distribution")
    figureTag = createModelTag(params) + "_readDistributionBefore.png"        
    fig1.savefig(os.path.join(modeloutputdirectory, figureTag))
    
    #select all rows with read count >= oversamplingPercentage of max read count and add them oversamplingFactor times to the df
    readMax = df.reads.max()
    cutPercentage = 0.5
    smallCountSamples = df[df['reads']<= cutPercentage*readMax].shape[0]
    highCountSamples = df[df['reads'] > cutPercentage*readMax].shape[0]
    sampleRelation = highCountSamples / smallCountSamples
    majorOversamplingFactor = 4
    while sampleRelation < majorOversamplingFactor:
        oversamplingPercentageList = np.linspace(cutPercentage,1.0,20)
        #compute oversampling factors
        totalNrSamples = df.shape[0]
        oversamplingFactorList = []
        for percentageLow, percentageHigh in zip(oversamplingPercentageList[0:-1], oversamplingPercentageList[1:]):
            gtMask = df['reads'] > percentageLow * readMax
            leqMask = df['reads'] <= percentageHigh * readMax
            oversampledDf = df[gtMask & leqMask]
            if not oversampledDf.empty:
                nrSamples = oversampledDf.shape[0]
                oversamplingFactorList.append(totalNrSamples / nrSamples) 
            else:
                oversamplingFactorList.append(np.inf)
        
        minOversamplingFactor = np.min(oversamplingFactorList)
        oversamplingFactorList = np.uint32(np.round(oversamplingFactorList / minOversamplingFactor))

        print("relation of 'values beyond threshold' / 'values below threshold': {0:.2f}".format(sampleRelation))
        for percentageLow, percentageHigh, factor in zip(oversamplingPercentageList[0:-1], oversamplingPercentageList[1:], oversamplingFactorList):
            gtMask = df['reads'] > percentageLow * readMax
            leqMask = df['reads'] <= percentageHigh * readMax
            oversampledDf = df[gtMask & leqMask]    
            if not oversampledDf.empty:
                msg = "{0:d} values are > {1:.2f} * max. read count and <= {2:.2f} * max. read count. Oversampling {3:d}x"
                msg = msg.format(oversampledDf.shape[0], percentageLow, percentageHigh, factor)
                print(msg)   
                appendDf = pd.concat([oversampledDf]*factor, ignore_index=True)
                df = df.append(appendDf, ignore_index=True, sort=False)
                df.reset_index(inplace=True, drop=True)
        
        highCountSamples = df[df['reads'] > 0.2*readMax].shape[0]
        sampleRelation = highCountSamples / smallCountSamples

    #plot the read distribution after oversampling
    fig1, ax1 = plt.subplots(figsize=(8,6))
    df.hist(column='reads', bins=100, ax=ax1, density=True)
    ax1.set_xlabel("read counts")
    ax1.set_ylabel("normalized frequency")
    ax1.set_title("Read distribution after variable oversampling\n(cutP: " + str(cutPercentage) + " factor: " + str(majorOversamplingFactor) + ")")
    figureTag = createModelTag(params) + "_readDistributionAfterVariableOversampling.png"        
    fig1.savefig(os.path.join(modeloutputdirectory, figureTag))
    
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
            raise NotImplementedError("unknown param for 'method'")
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
