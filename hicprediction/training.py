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
import pandas as pd
from sklearn.tree import export_graphviz
import pydot
import math

""" Module responsible for the training of the regressor with data sets.
"""
@conf.train_options
@click.command()
def train(modeloutputdirectory, conversion, trees, maxfeat, traindatasetfile, nodist, nomiddle, nostartend, ovspercentage, ovsfactor, ovsbalance):
    """
    Wrapper function for click
    """
    if maxfeat == 'none':
        maxfeat = None
    training(modeloutputdirectory, conversion, trees, maxfeat, traindatasetfile, nodist, nomiddle, nostartend, ovspercentage, ovsfactor, ovsbalance)

def training(modeloutputdirectory, conversion, pNrOfTrees, pMaxFeat, traindatasetfile, noDist, noMiddle, noStartEnd, pOvsPercentage, pOvsFactor, pOvsBalance):
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

    ### drop columns that should not be used for training EXCEPT reads which are needed for oversampling
    ### dropping also reduces the memory demand for subsequent oversampling
    dropList = ['first', 'second', 'chrom', 'avgRead']
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
    X = df.drop(columns=dropList)

    ### oversampling to emphasize data with high read counts
    if pOvsPercentage > 0.0 and pOvsPercentage < 1.0 and pOvsFactor > 0.0:
        variableOversampling(X, params, pCutPercentage=pOvsPercentage, pOversamplingFactor=pOvsFactor, pBalance=pOvsBalance, pModeloutputdirectory=modeloutputdirectory, pPlotOutput=True)
    
    ### separate reads from data and convert, if necessary
    if conversion == 'none':
        y = X['reads']
    elif conversion == 'standardLog':
        y = np.log(X['reads']+1)
    X.drop(columns='reads', inplace=True)

    model.fit(X, y)
    modelTag = createModelTag(params)
    modelFileName = os.path.join(modeloutputdirectory, modelTag + ".z")
    joblib.dump((model, params), modelFileName, compress=True ) 

    visualizeModel(model, modeloutputdirectory, list(X.columns), modelTag)


def variableOversampling(pInOutDataFrameWithReads, pParams, pCutPercentage=0.2, pOversamplingFactor=4.0, pBalance=False, pModeloutputdirectory=None, pPlotOutput=False):
    #Select all rows (samples) from dataframe where "reads" (read counts) are in certain range,
    #i.e. are higher than pCutPercentage*maxReadCount. This is the high-read-count-range.
    #Emphasize this range by adding (copying) the respective rows to the dataframe such that the number of high-read-count rows 
    #in the dataframe is at least pOversamplingFactor times greater than the number of low-read-count samples
    #Additionally, the high-read-count-range can be approximately balanced in itself by binning it into regions 
    #and emphasizing regions with lower numbers of samples more than the ones with higher number of samples.
    print("Oversampling...")
    if pModeloutputdirectory and pPlotOutput:
        #plot the raw read distribution first
        fig1, ax1 = plt.subplots(figsize=(8,6))
        pInOutDataFrameWithReads.hist(column='reads', bins=100, ax=ax1, density=True) 
        ax1.set_xlabel("read counts")
        ax1.set_ylabel("normalized frequency")
        ax1.set_title("raw read distribution")
        figureTag = createModelTag(pParams) + "_readDistributionBefore.png"        
        fig1.savefig(os.path.join(pModeloutputdirectory, figureTag))
    
    readMax = np.float32(pInOutDataFrameWithReads.reads.max())
    #the range which doesn't get oversampled
    smallCountSamples = pInOutDataFrameWithReads[pInOutDataFrameWithReads['reads']<= pCutPercentage*readMax].shape[0]
    if smallCountSamples == 0:
        print("Warning: no samples with read count less than {:.2f}*max read count in dataset".format(pCutPercentage))
        print("No oversampling possible, continuing with next step")
        print("Consider increasing oversamling percentage using -ovsP parameter")
        return
    #the range we want to oversample, i.e. add repeatedly to the dataframe until sampleRelation >= pOversamplingFactor
    highCountSamples = pInOutDataFrameWithReads[pInOutDataFrameWithReads['reads'] > pCutPercentage*readMax].shape[0]
    sampleRelation = highCountSamples / smallCountSamples
    print("=> relation high-read-count-range :: low-read-count-range before oversampling {:.2f}".format(sampleRelation))
        
    #if enabled, first balance the high-count samples among each other, i. e. bin them and oversample the bins
    if pBalance and sampleRelation < pOversamplingFactor:
        print("=> balancing the high-read-count samples among each other")
        oversamplingPercentageList = np.linspace(pCutPercentage,1.0,30)
        oversamplingFactorList = []
        oversamlingDfList= []
        #compute the balancing factors
        for percentageLow, percentageHigh in zip(oversamplingPercentageList[0:-1], oversamplingPercentageList[1:]):
            gtMask = pInOutDataFrameWithReads['reads'] > percentageLow * readMax
            leqMask = pInOutDataFrameWithReads['reads'] <= percentageHigh * readMax
            oversampledDf = pInOutDataFrameWithReads[gtMask & leqMask]
            if not oversampledDf.empty:
                oversamlingDfList.append(oversampledDf) 
                nrOfSamplesInBin = oversampledDf.shape[0]
                oversamplingFactorList.append(nrOfSamplesInBin)
        maxNrSamplesInBins = np.max(oversamplingFactorList)
        oversamplingFactorList = np.uint32([np.round(maxNrSamplesInBins/x) - 1 for x in oversamplingFactorList])    

        #balance the high-read-count range by appending the individual bins factor times to the original dataframe
        #the bin with the highest number will not be appended
        for oversampledDf, factor in zip(oversamlingDfList, oversamplingFactorList): 
            if not factor < 1:
                appendDf = pd.concat([oversampledDf]*factor, ignore_index=True)
                pInOutDataFrameWithReads = pInOutDataFrameWithReads.append(appendDf, ignore_index=True, sort=False)
                pInOutDataFrameWithReads.reset_index(inplace=True, drop=True)
        
        #recompute the relation
        highCountSamples = pInOutDataFrameWithReads[pInOutDataFrameWithReads['reads'] > pCutPercentage*readMax].shape[0]
        sampleRelation = highCountSamples / smallCountSamples
        print("=> relation high-read-count-range :: low-read-count-range after balancing {:.2f}".format(sampleRelation))

    #if the relation between the samples in both ranges is (still) below the desired factor, proceed with oversampling
    if sampleRelation < pOversamplingFactor:
        factor = math.ceil(pOversamplingFactor / sampleRelation)
        gtMask = pInOutDataFrameWithReads['reads'] > pCutPercentage * readMax
        oversampledDf = pInOutDataFrameWithReads[gtMask]
        appendDf = pd.concat([oversampledDf]*factor)
        pInOutDataFrameWithReads = pInOutDataFrameWithReads.append(appendDf, ignore_index=True, sort=False)
        pInOutDataFrameWithReads.reset_index(inplace=True, drop=True)
        #recompute for printing the final result
        highCountSamples = pInOutDataFrameWithReads[pInOutDataFrameWithReads['reads'] > pCutPercentage*readMax].shape[0]
        sampleRelation = highCountSamples / smallCountSamples
        print("=> relation high-read-count-range :: low-read-count-range after oversampling {:.2f}".format(sampleRelation))
        print("oversampling done with factor {:d}".format(factor))

    if pModeloutputdirectory and pPlotOutput:
        #plot the read distribution after oversampling
        fig1, ax1 = plt.subplots(figsize=(8,6))
        pInOutDataFrameWithReads.hist(column='reads', bins=100, ax=ax1, density=True)
        ax1.set_xlabel("read counts")
        ax1.set_ylabel("normalized frequency")
        ax1.set_title("Read distribution after oversampling\n(cutP: {:.2f}, factor: {:.2f}, balanced: {})".format(pCutPercentage, float(pOversamplingFactor), pBalance))
        figureTag = createModelTag(pParams) + "_readDistributionAfterVariableOversampling.png"        
        fig1.savefig(os.path.join(pModeloutputdirectory, figureTag))

def visualizeModel(pTreeBasedLearningModel, pOutDir, pFeatList, pModelTag):
    #plot visualizations of the trees to png files
    #only up to depth 6 to keep memory demand low and image "readable"
    #compare this tutorial: https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
    print("plotting trees...")
    for treeNr, tree in enumerate(pTreeBasedLearningModel.estimators_):
        treeStrDot = pModelTag + "_tree" + str(treeNr + 1) + ".dot"
        treeStrPng = pModelTag + "_tree" + str(treeNr + 1) + ".png"
        export_graphviz(tree, out_file = os.path.join(pOutDir, treeStrDot), max_depth=6, feature_names=pFeatList, rounded = True, precision = 6)
        (graph, ) = pydot.graph_from_dot_file(os.path.join(pOutDir,treeStrDot))
        graph.write_png(treeStrPng)
        os.remove(os.path.join(pOutDir, treeStrDot))

    #print and plot feature importances
    #compare https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
    importances = pTreeBasedLearningModel.feature_importances_
    std = np.std([tree.feature_importances_ for tree in pTreeBasedLearningModel.estimators_],
             axis=0)
    indices = np.argsort(importances)[::-1]
    if len(indices) != len(pFeatList):
        msg = "Aborting while visualizing the model\n"
        msg += "Number of feature names doesn't match number of features"
        raise ValueError(msg)

    print()
    print("Feature importances:")
    for f in range(len(pFeatList)):
        print("{0:d}. feature {1:d} ({2:s}; {3:.2f})".format(f + 1, indices[f], pFeatList[indices[f]], importances[indices[f]]))
    print()

    nrTrees = len(pTreeBasedLearningModel.estimators_)
    imgWidth = max(5, np.round(len(indices)/7))
    imgHeight = 0.75*imgWidth
    fig1, ax1 = plt.subplots(constrained_layout=True, figsize=(imgWidth,imgHeight))
    ax1.set_title("Feature importances and std deviations ({:d} trees)".format(nrTrees))
    ax1.bar(2*indices, importances[indices],
            color="r", yerr=std[indices], align="center", width=1.5)
    ax1.set_xticks(2*indices)
    ax1.set_xticklabels(np.array(pFeatList)[indices], rotation=90, fontsize=6)
    ax1.set_ylim([0.0,1.0]) #allow comparing results from different params, datasets etc.
    ax1.set_yticks(np.linspace(0,1,11))
    ax1.set_xlabel("feature name")
    ax1.set_ylabel("relative feature importance")
    importanceFigStr = pModelTag + "_importanceGraph.png"
    fig1.savefig(os.path.join(pOutDir, importanceFigStr))

if __name__ == '__main__':
    train()
