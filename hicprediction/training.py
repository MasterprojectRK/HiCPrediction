#!/usr/bin/env python3
import os
os.environ['NUMEXPR_MAX_THREADS'] = '16'
os.environ['NUMEXPR_NUM_THREADS'] = '8'
import hicprediction.configurations as conf
import click
import joblib
import sklearn.ensemble
from sklearn.model_selection import KFold
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
@click.command(context_settings=dict(
                                    ignore_unknown_options=True,
                                    allow_extra_args=True))
@click.pass_context
def train(ctx, modeloutputdirectory, conversion, traindatasetfile, nodist, nomiddle, nostartend, ovspercentage, ovsfactor, ovsbalance, plottrees, splittrainset, useextratrees):
    """
    Wrapper function for click
    """

    #default parameters for the random forest (trying to match HiC-Reg)
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
    #allowed data types, if someone wanted to set random forest parameters
    # different from the defaults above  
    allowedModelParamDytpeDict = dict(n_estimators=[int], 
                            criterion=[str], 
                            max_depth=[int], 
                            min_samples_split=[int,float], 
                            min_samples_leaf=[int,float], 
                            min_weight_fraction_leaf=[float], 
                            max_features=[int,float,str], 
                            max_leaf_nodes=[int], 
                            min_impurity_decrease=[float], 
                            min_impurity_split=[float], 
                            bootstrap=[bool], 
                            oob_score=[bool], 
                            n_jobs=[int], 
                            random_state=[int], 
                            verbose=[int], 
                            ccp_alpha=[float], 
                            max_samples=[int,float]  )
        #extra trees should sample without replacement = no bootstrapping by default
    if useextratrees:
        modelParamDict['bootstrap'] = False

    #try to extract additional parameters for the random forest / extra trees algorithm
    for i in range(0, len(ctx.args), 2):
        paramName= ctx.args[i][2:]
        if paramName in modelParamDict:
            param = tryConvert(ctx.args[i+1], allowedModelParamDytpeDict[paramName])
            modelParamDict[paramName]=param
        else:
            msg = "Parameter {:s} is not supported".format(paramName)
            raise SystemExit(msg)

    training(pModeloutputdirectory = modeloutputdirectory, \
                pConversion=conversion, \
                pModelParamDict=modelParamDict, \
                pTraindatasetfile=traindatasetfile,\
                pNoDist = nodist,\
                pNoMiddle = nomiddle,\
                pNoStartEnd= nostartend,\
                pWeightBound1= weightbound1,\
                pWeightBound2 = weightbound2,\
                pOvsFactor = ovsfactor,\
                pOvsBalance = ovsbalance,\
                pPlotTrees = plottrees,\
                pSplitTrainset = splittrainset,\
                pUseExtraTrees = useextratrees)

def training(pModeloutputdirectory, pConversion, pModelParamDict, pTraindatasetfile, pNoDist, pNoMiddle, pNoStartEnd, pWeightBound1, pWeightBound2, pOvsFactor, pOvsBalance, pPlotTrees, pSplitTrainset, pUseExtraTrees):
    """
    Train function
    Attributes:
        modeloutputdirectory -- path to desired store location of models
        conversion --  read values conversion method 
        traindatasetfile -- input data set for training
    """

    ### load dataset and set parameters
    df, params, msg = checkTrainingset(pTraindatasetfile, pKeepInvalid=True, pCheckReads=True)
    if msg != None:
        msg = "Aborting.\n" + msg
        raise SystemExit(msg)
    
    ### store the training parameters for later reference and file name creation
    params['conversion'] = pConversion
    params['noDistance'] = pNoDist
    params['noMiddle'] = pNoMiddle
    params['noStartEnd'] = pNoStartEnd
    params['useExtraTrees'] = pUseExtraTrees
        
    ### create model with desired parameters
    try:
        if pUseExtraTrees:
            model = sklearn.ensemble.ExtraTreesRegressor(**pModelParamDict)
            print("using extra trees regressor")
        else:
            model = sklearn.ensemble.RandomForestRegressor(**pModelParamDict)
            print("using random forest regressor")
        params['learningParams'] = pModelParamDict
    except Exception as e:
        msg = str(e) + "\n"
        msg += "Could not create model. Check parameters"
        raise SystemExit(msg)

    ### replace infinity and nan values. There shouldn't be any.
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    if not df[df.isna().any(axis=1)].empty:
        msg = "Warning: There are {:d} rows in the training which contain NaN\n"
        msg = msg.format(df[df.isna().any(axis=1)].shape[0])
        msg += "The NaNs are in column(s) {:s}\n"
        msg = msg.format(", ".join(df[df.isna().any(axis=1)].columns))
        msg += "Replacing by zeros. Check input data!"
        print(msg)
        df.fillna(value=0, inplace=True)

    ### add weights and over-emphasize some of them, if requested
    variableOversampling(df, params, pOvsPercentage, pOvsFactor, pOvsBalance, modeloutputdirectory, pPlotOutput=True)

    ### drop columns that should not be used for training
    dropList = ['first', 'second', 'chrom', 'reads', 'avgRead', 'valid', 'weights']
    if pNoDist:
        dropList.append('distance')    
    if pNoMiddle:
        if params['method'] == 'oneHot':
            dropList.append('middleProt')
        elif params['method'] == 'multiColumn':
            numberOfProteins = int((df.shape[1] - 6) / 3)
            for protein in range(numberOfProteins):
                dropList.append(str(protein + numberOfProteins))
        else:
            raise NotImplementedError("unknown param for 'method'")
    if pNoStartEnd:
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
    if pConversion == 'none':
        y = df['reads']
    elif pConversion == 'standardLog':
        y = np.log(df['reads']+1)

    ## train models and store them
    if pSplitTrainset:
        #as proposed by Zhang et al.,
        #split datsets into 5 folds for cross validation
        #and train a model from each
        #later, prediction can be averaged from these 5 models
        kfoldCV = KFold(n_splits = 5, \
                        shuffle = True, \
                        random_state = 35)
        for i, (trainIndices, testIndices) in enumerate(kfoldCV.split(X)):
            X_train = X.iloc[trainIndices,:]
            y_train = y.iloc[trainIndices]
            weights_train = weights.iloc[trainIndices]
            X_test = X.iloc[testIndices,:]
            y_test = y.iloc[testIndices]
            weights_test = weights.iloc[testIndices]
            model.fit(X_train, y_train, weights_train)
            trainScore = model.score(X_train, y_train, weights_train)
            testScore = model.score(X_test, y_test, weights_test)
            modelTag = createModelTag(params) + "_" + str(i+1)
            modelFileName = os.path.join(pModeloutputdirectory, modelTag + ".z")
            joblib.dump((model, params), modelFileName, compress=True ) 
            print("processed model {:d}".format(i+1))
            print("trainScore:", trainScore)
            print("testScore:", testScore)
            featNamesList = createNamesForFeatures(list(X_train.columns), params)
            visualizeModel(model, pModeloutputdirectory, featNamesList, modelTag, pPlotTrees)
    else:
        model.fit(X, y, weights)
        modelTag = createModelTag(params)
        modelFileName = os.path.join(pModeloutputdirectory, modelTag + ".z")
        joblib.dump((model, params), modelFileName, compress=True )
        featNamesList = createNamesForFeatures(list(X.columns), params)
        visualizeModel(model, pModeloutputdirectory, featNamesList, modelTag, pPlotTrees)

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

def visualizeModel(pTreeBasedLearningModel, pOutDir, pFeatList, pModelTag, pPlotTrees):
    #plot visualizations of the trees to png files
    #only up to depth 6 to keep memory demand low and image "readable"
    #compare this tutorial: https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
    if pPlotTrees:
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
    columnWidth = 1.5
    ax1.bar(2*indices, importances[indices],
            color="r", yerr=std[indices], align="center", width=columnWidth)
    ax1.set_xticks(2*indices)
    ax1.set_xlim([min(2*indices)-columnWidth, max(2*indices)+columnWidth])
    ax1.set_xticklabels(np.array(pFeatList)[indices], rotation=90, fontsize=6)
    ax1.set_ylim([0.0,1.0]) #allow comparing results from different params, datasets etc.
    ax1.set_yticks(np.linspace(0,1,11))
    ax1.set_xlabel("feature name")
    ax1.set_ylabel("relative feature importance")
    importanceFigStr = pModelTag + "_importanceGraph.pdf"
    fig1.savefig(os.path.join(pOutDir, importanceFigStr))

def tryConvert(pParamStr, pAllowedTypeList):
    converted = False
    returnParam = str(pParamStr)
    if pAllowedTypeList and isinstance(pAllowedTypeList, list) and len(pAllowedTypeList) > 0:
        i = 0
        while i < len(pAllowedTypeList) and not converted:
            try:
                returnParam = pAllowedTypeList[i](returnParam)
                converted = True

            except:
                converted = False
            i += 1
    return returnParam

def createNamesForFeatures(pDataframeColumnList, pParams):
    #check if all relevant parameters exist, 
    #if not return input list back
    if not 'proteinFileNames' in pParams  \
       or not 'noDistance' in pParams \
       or not 'noMiddle' in pParams \
       or not 'noStartEnd' in pParams \
       or not 'method' in pParams:
        return pDataframeColumnList
    
    if pParams['method'] != 'multiColumn':
        return pDataframeColumnList #not handled for now

    startFeatList = [x + '.start' for x in pParams['proteinFileNames']]
    endFeatList = [x + '.end' for x in pParams['proteinFileNames']]
    windowFeatList = [x + '.window' for x in pParams['proteinFileNames']]

    returnList = []
    if not pParams['noStartEnd'] == True:
        returnList += startFeatList
    if not pParams['noMiddle'] == True:
        returnList += windowFeatList
    if not pParams['noStartEnd'] == True:
        returnList += endFeatList
    if not pParams['noDistance'] == True:
        returnList += ['distance']

    if not len(returnList) == len(pDataframeColumnList):
        #some feature must be missing, so return original list
        #this shouldn't happen
        returnList = pDataframeColumnList 
    
    return returnList
     

def checkTrainingset(pTrainDatasetFile, pKeepInvalid=False, pCheckReads=True):
    msg = ""
    trainDf = None
    paramsDict = None
    try:
        trainDf, paramsDict = joblib.load(pTrainDatasetFile)
    except Exception as e:
        msg += "Failed loading dataset. Wrong format?\n"
        msg += "Original error message " + str(e) + "\n"
        return trainDf, paramsDict, msg
    if not isinstance(trainDf, pd.DataFrame):
        msg += "Input {:s} is not a dataset\n"
        if isinstance(trainDf, sklearn.ensemble.forest.ForestRegressor):
            msg += "Maybe a trained model was entered instead of a dataset?"
        msg = msg.format(pTrainDatasetFile)
    else:
        if not pKeepInvalid:
            validMask = trainDf['valid'] == True
            trainDf = trainDf[validMask]
        if trainDf.empty:
            msg += "Training dataset is empty or contains only invalid samples"
        if pCheckReads:
            ### check if the dataset contains reads, otherwise it cannot be used for training
            nanreadMask = trainDf['reads'] == np.nan
            if not trainDf[nanreadMask].empty:
                #any nans coming from bad bins in the HiC-matrix must be replaced when creating the dataset
                #so there should not be any nans left
                msg += "dataset {0:s} cannot be used for training" 
                msg += "because it contains no target reads"
                msg = msg.format(pTrainDatasetFile)
    if msg == "":
        msg = None
    return trainDf, paramsDict, msg
        

if __name__ == '__main__':
    train() # pylint: disable=no-value-for-parameter
