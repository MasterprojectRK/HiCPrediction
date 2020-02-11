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
def train(modeloutputdirectory, conversion, trees, maxfeat, traindatasetfile, taddomainfile, nodist, nomiddle, nostartend, ovspercentage, ovsfactor, ovsbalance):
    """
    Wrapper function for click
    """
    if maxfeat == 'none':
        maxfeat = None
    training(modeloutputdirectory, conversion, trees, maxfeat, traindatasetfile, taddomainfile, nodist, nomiddle, nostartend, ovspercentage, ovsfactor, ovsbalance)

def training(modeloutputdirectory, conversion, pNrOfTrees, pMaxFeat, traindatasetfile, taddomainfile, noDist, noMiddle, noStartEnd, pOvsPercentage, pOvsFactor, pOvsBalance):
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
    try:
        df, params = joblib.load(traindatasetfile)
    except Exception as e:
        print(e)
        msg = "Failed loading dataset. Wrong format?"
        sys.exit(msg)
    if not isinstance(df, pd.DataFrame):
        msg = "Aborting. Input {:s} is not a dataset\n"
        if isinstance(df, sklearn.ensemble.forest.ForestRegressor):
            msg += "Maybe a trained model was entered instead of a dataset?"
        msg = msg.format(traindatasetfile)
        sys.exit(msg)
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
    model = sklearn.ensemble.ExtraTreesRegressor(max_features=pMaxFeat, random_state=5,\
                    n_estimators=pNrOfTrees, n_jobs=-1, verbose=2, criterion='mse')
    
    ### replace infinity and nan values. There shouldn't be any.
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(value=0, inplace=True)

    ### add weights and over-emphasize some of them, if requested
    #variableOversampling(df, params, pOvsPercentage, pOvsFactor, pOvsBalance, modeloutputdirectory, pPlotOutput=True)
    tadOversampling(df, params, taddomainfile, pOvsFactor)

    ### drop columns that should not be used for training
    dropList = ['first', 'second', 'chrom', 'reads', 'avgRead', 'weights']
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
    X = df.drop(columns=dropList, errors='ignore')

    #weights
    weights = df['weights']

    ### apply conversion
    if conversion == 'none':
        y = df['reads']
    elif conversion == 'standardLog':
        y = np.log(df['reads']+1)

    ## train model and store it
    model.fit(X, y, weights)
    modelTag = createModelTag(params)
    modelFileName = os.path.join(modeloutputdirectory, modelTag + ".z")
    joblib.dump((model, params), modelFileName, compress=True ) 
    print("\n")

    visualizeModel(model, modeloutputdirectory, list(X.columns), modelTag)


def variableOversampling(pInOutDataFrameWithReads, pParams, pCutPercentage=0.2, pOversamplingFactor=4.0, pBalance=False, pModeloutputdirectory=None, pPlotOutput=False):
    #Select all rows (samples) from dataframe where "reads" (read counts) are in certain range,
    #i.e. are higher than pCutPercentage*maxReadCount. This is the high-read-count-range.
    #Emphasize this range by adjusting the weights such that (sum of high-read-count-weights)/(sum of low-read-count-weights) = pOversamplingFactor
    #Additionally, the high-read-count-range can be approximately balanced in itself by binning it into regions 
    #and emphasizing regions with lower numbers of samples more than the ones with higher number of samples.
    
    #if nothing works, all weights equally = 1.0
    pInOutDataFrameWithReads['weights'] = 1.0

    if pCutPercentage <= 0.0 or pCutPercentage >= 1.0 or pOversamplingFactor <= 0.0:
        return

    readMax = np.float32(pInOutDataFrameWithReads.reads.max())
    print("Oversampling...")
    if pModeloutputdirectory and pPlotOutput:
        #plot the weight vs. read count distribution first
        fig1, ax1 = plt.subplots(figsize=(8,6))
        binSize = readMax / 100
        bins = pd.cut(pInOutDataFrameWithReads['reads'], bins=np.arange(0,readMax + binSize, binSize), labels=np.arange(0,readMax,binSize), include_lowest=True)
        printDf = pInOutDataFrameWithReads[['weights']].groupby(bins).sum()
        printDf.reset_index(inplace=True, drop=False)
        ax1.bar(printDf.reads, printDf.weights, width=binSize)
        ax1.set_xlabel("read counts")
        ax1.set_ylabel("weight sum")
        ax1.set_yscale('log')
        ax1.set_title("weight sum vs. read counts")
        figureTag = createModelTag(pParams) + "_weightSumDistributionBefore.png"        
        fig1.savefig(os.path.join(pModeloutputdirectory, figureTag))
    
    
    #the range which doesn't get oversampled
    smallCountWeightSum = pInOutDataFrameWithReads[pInOutDataFrameWithReads['reads']<= pCutPercentage*readMax].weights.sum()
    if smallCountWeightSum == 0:
        print("Warning: no samples with read count less than {:.2f}*max read count in dataset".format(pCutPercentage))
        print("No oversampling possible, continuing with next step and all weights = 1.0")
        print("Consider increasing oversamling percentage using -ovsP parameter")
        return
    #the range we want to oversample, i.e. adjust the weights
    highCountWeightSum = pInOutDataFrameWithReads[pInOutDataFrameWithReads['reads'] > pCutPercentage*readMax].weights.sum()
    sampleRelation = highCountWeightSum / smallCountWeightSum
    print("=> weight sum relation high-read-count-range :: low-read-count-range before oversampling {:.2f}".format(sampleRelation))
    
    #if balancing enabled, split the high-count samples into a reasonable number of bins, 
    #then adjust the weights for the bins separately
    #such that all bins have the same weight sum and all samples within each bin have the same weight
    #if balancing disabled => corner case with just one bin
    if pBalance:
        binsizePercent = 0.025
        nrOfBins = math.ceil( (1.0-pCutPercentage) / binsizePercent)
    else:
        nrOfBins = 1 
    msg = "=> splitting the the high-read-count range into {:d} bins and adjusting weights"
    msg = msg.format(nrOfBins)
    print(msg)
    oversamplingPercentageList = np.linspace(pCutPercentage,1.0, nrOfBins + 1)
    #find the number of bins which have reads in them and compute the target weight sum
    #at least one bin will have reads
    nrOfNonzeroBins = nrOfBins
    for percentageLow, percentageHigh in zip(oversamplingPercentageList[0:-1], oversamplingPercentageList[1:]):
        gtMask = pInOutDataFrameWithReads['reads'] > percentageLow * readMax
        leqMask = pInOutDataFrameWithReads['reads'] <= percentageHigh * readMax
        if pInOutDataFrameWithReads[gtMask & leqMask].empty:
            nrOfNonzeroBins -= 1
    targetWeightSum = pOversamplingFactor * smallCountWeightSum / nrOfNonzeroBins #target weight sum for each bin
    #adjust the weights for each bin separately
    for percentageLow, percentageHigh in zip(oversamplingPercentageList[0:-1], oversamplingPercentageList[1:]):
        gtMask = pInOutDataFrameWithReads['reads'] > percentageLow * readMax
        leqMask = pInOutDataFrameWithReads['reads'] <= percentageHigh * readMax
        if not pInOutDataFrameWithReads[gtMask & leqMask].empty:
            currentNrOfSamples = pInOutDataFrameWithReads[gtMask & leqMask].shape[0]
            pInOutDataFrameWithReads.loc[gtMask & leqMask, 'weights'] = targetWeightSum / currentNrOfSamples     

    #recompute and print the new weight sums
    highCountWeightSum = pInOutDataFrameWithReads[pInOutDataFrameWithReads['reads'] > pCutPercentage*readMax].weights.sum()
    msg = "=> Sum of weights in low-read-count range (<={:.2f}*max. read count) is now {:.2f}\n".format(pCutPercentage, smallCountWeightSum)
    msg += "=> Sum of weights in high-read-count range (>{:.2f}*max. read count) is now {:.2f}\n".format(pCutPercentage, highCountWeightSum)
    msg += "=> weight factor = {:.2f}\n".format(highCountWeightSum/smallCountWeightSum)
    print(msg)

    if pModeloutputdirectory and pPlotOutput:
        #plot the weight vs. read count distribution after oversampling
        binSize = readMax / 100
        bins = pd.cut(pInOutDataFrameWithReads['reads'], bins=np.arange(0,readMax + binSize, binSize), labels=np.arange(0,readMax,binSize), include_lowest=True)
        printDf = pInOutDataFrameWithReads[['weights']].groupby(bins).sum()
        printDf.reset_index(inplace=True, drop=False)
        ax1.bar(printDf.reads, printDf.weights, width=binSize)
        ax1.set_xlabel("read counts")
        ax1.set_ylabel("weight sum")
        ax1.set_yscale('log')
        ax1.set_title("weight sum vs. read counts")
        figureTag = createModelTag(pParams) + "_weightSumDistributionAfter.png"        
        fig1.savefig(os.path.join(pModeloutputdirectory, figureTag))


def tadOversampling(pInOutDataFrameWithReads, pParams, pTadDomainFile, pOvsFactor, pMaxDistance=500000):
    #if nothing works, all weights equally = 1.0
    pInOutDataFrameWithReads['weights'] = 1.0

    if not pTadDomainFile or pOvsFactor == 0: 
        return

    try:
        bedColumnNames = ["chrom", "chromStart", "chromEnd", "name", 
                        "score", "strand", "thickStart", "thickEnd" , "itemRgb", "blockCount", "blockSizes", "blockStarts"]
        tadDomainDf = pd.read_csv(pTadDomainFile, sep="\t", header=0)
        tadDomainDf.columns = bedColumnNames[0:tadDomainDf.shape[1]]
        chromMask = tadDomainDf['chrom'] == pParams['chrom']
        tadDomainDf = tadDomainDf[chromMask]
    except Exception as e:
        msg = "Ignoring invalid tadDomainFile."
        print(e, msg)
        return

    if tadDomainDf.empty:
        msg = "Warning: tadDomainFile does not contain data for chromosome {:s}\n"
        msg += "Ignoring TADs"
        msg = msg.format(str(pParams['chrom']))
        print(msg)
        return
    
    for start, end in zip(tadDomainDf['chromStart'], tadDomainDf['chromEnd']):
        #mark the samples which are within TADs by setting weight to zero first
        if end - start < pMaxDistance: #prediction is often quite good for medium sized structures, so no need to overweight them
            startMask = pInOutDataFrameWithReads['first'] >= math.floor(start / int(pParams['resolution']))
            endMask = pInOutDataFrameWithReads['second'] <= math.floor(end / int(pParams['resolution']))
            pInOutDataFrameWithReads.loc[ startMask & endMask, 'weights'] = 0 

    #overweight the samples inside TADs (marked by weight=0) as specified by pOvsFactor
    outsideTadMask = pInOutDataFrameWithReads['weights'] > 0
    insideTadMask = pInOutDataFrameWithReads['weights'] == 0
    weightOutsideTads = pInOutDataFrameWithReads[outsideTadMask].weights.sum()
    targetWeightInsideTads = weightOutsideTads * pOvsFactor
    nrOfSamplesInsideTads = pInOutDataFrameWithReads[insideTadMask].shape[0]
    if nrOfSamplesInsideTads > 0:
        pInOutDataFrameWithReads.loc[insideTadMask, 'weights'] = targetWeightInsideTads / pInOutDataFrameWithReads[insideTadMask].shape[0]
        weightInsideTads = pInOutDataFrameWithReads[insideTadMask].weights.sum()
        print("Oversampling:")
        print("Sum of weights outside TADs: {:.0f}".format(weightOutsideTads))
        print("Sum of weights inside TADs: {:.0f}".format(weightInsideTads))
        print("Weight factor {:.2f}".format(weightInsideTads / weightOutsideTads))
    else:
        pInOutDataFrameWithReads['weights'] = 1.0
        msg = "No samples inside provided TADs found. This shouldn't happen.\n"
        msg += "Check your TAD domain file, maybe from wrong cell line? \n"
        msg += "Cell line of dataset is {:s}\n".format(str(pParams['cellType']))
        msg += "Ignoring TADs and setting weights = 1.0"
        print(msg)


def visualizeModel(pTreeBasedLearningModel, pOutDir, pFeatList, pModelTag):
    #plot visualizations of the trees to png files
    #only up to depth 6 to keep memory demand low and image "readable"
    #compare this tutorial: https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
    print("plotting trees...")
    for treeNr, tree in enumerate(pTreeBasedLearningModel.estimators_):
        treeStrDot = pModelTag + "_tree" + str(treeNr + 1) + ".dot"
        treeStrPng = pModelTag + "_tree" + str(treeNr + 1) + ".png"
        dotfile = os.path.join(pOutDir, treeStrDot)
        pngfile = os.path.join(pOutDir, treeStrPng)
        export_graphviz(tree, out_file = dotfile, max_depth=6, feature_names=pFeatList, rounded = True, precision = 6)
        (graph, ) = pydot.graph_from_dot_file(dotfile)
        graph.write_png(pngfile)
        os.remove(dotfile)

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
