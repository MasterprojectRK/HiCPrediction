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
from hicprediction.utilities import createModelTag
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import export_graphviz
import pydot
import math

""" Module responsible for the training of the regressor with data sets.
"""
@conf.train_options
@click.version_option()
@click.command(context_settings=dict(
                                    ignore_unknown_options=True,
                                    allow_extra_args=True))
@click.pass_context
def train(ctx, modeloutputdirectory, conversion, traindatasetfile, 
            nodist, nomiddle, nostartend, weightbound1, weightbound2, 
            weightingtype, featlist, ovsfactor, taddomainfile,
            plottrees, splittrainset, useextratrees):

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
    #allowed data types, if someone wanted to set other random forest parameters
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
                pWeightingType=weightingtype, \
                pFeatList=featlist, \
                pOvsFactor = ovsfactor,\
                pTadDomainFile = taddomainfile,\
                pPlotTrees = plottrees,\
                pSplitTrainset = splittrainset,\
                pUseExtraTrees = useextratrees)

def training(pModeloutputdirectory, pConversion, pModelParamDict, 
                pTraindatasetfile, pNoDist, pNoMiddle, pNoStartEnd,
                pWeightBound1, pWeightBound2, pWeightingType, 
                pFeatList, pOvsFactor, pTadDomainFile, pPlotTrees, 
                pSplitTrainset, pUseExtraTrees):
    """
    Train function
    Attributes:
        modeloutputdirectory -- str, path to desired store location of models
        conversion --  str, read values conversion method 
        modelParamDict -- dictionary with all options for random forest and extra trees
        traindatasetfile -- str, input data set for training
        noDist -- bool, do not use distance for training
        noMiddle -- bool, do not use Window features for training
        noStartEnd -- bool, do not use start- and end features for training
        weightBound1 -- numeric upper or lower boundary for weighting samples
        weightBound2 -- numeric upper or lower boundary for weighting samples
        weightingType -- str, weight samples based on reads, protein Features or TADs
        featList -- list of str, names of features for sample weighting
        ovsFactor -- factor F for sample weighting such that weightSum_weightedSamples/weightSum_unweightedSamples = F
        plotTrees -- plot resulting trees down to level 6 (slow!)
        splitTrainset -- perform 5-fold cross validation split of dataset and train 5 models independently
        useExtratrees -- use extra trees algorithm instead of random forests (includes bootstrap=False)
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

    ### add weights and over-emphasize some samples, if requested
    columnList = None
    featureBasedWeigthing = False
    weightingSuccess = False
    if pWeightingType == 'proteinFeatures':
            columnList = [str(i) for i in pFeatList]
            featureBasedWeigthing = True
    elif pWeightingType == 'reads':
        columnList = ['reads']
        featureBasedWeigthing = True
    else: #TAD based weighting
        weightingSuccess = tadWeighting(pInOutDataFrame=df,
                            pParams=params, 
                            pTadDomainFile=pTadDomainFile,
                            pFactorFloat=pOvsFactor)
    if featureBasedWeigthing: #based on reads or protein features
        weightingSuccess = computeWeights(pDataframe=df, 
                                pBound1=pWeightBound1, 
                                pBound2=pWeightBound2, 
                                pFactorFloat=pOvsFactor, 
                                pParams=params, 
                                pColumnsToUse=columnList)
    if weightingSuccess == False:
        msg = "All sample weights set to 1"
        print(msg)
    
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

def computeWeights(pDataframe, pBound1, pBound2, pFactorFloat, pParams, pColumnsToUse=['reads']):
    ### weight all samples in pDataframe where the value of the columns 
    ### in the pColumnsToUse list
    ### is between pBound1 and pBound2.
    ### Weighting will be done such that the weight sum of all samples 
    ### between the bounds is pFactorFloat-times the weight sum 
    ### of all samples outside the bounds
    ### integer weights will be added added to pDataframe as a separate column 'weights'

    #equal weights for all samples as a start
    pDataframe['weights'] = 1 
    success = False
    #weight computation for oneHot method and factors <= 0 not supported
    if not 'method' in pParams or pParams['method'] != 'multiColumn' or pFactorFloat <= 0.0:
        return False
    if not pColumnsToUse or len(pColumnsToUse) == 0:
        return False
    #all feature columns / read column must be present in df
    for element in pColumnsToUse:
        if element not in pDataframe.columns:
            msg = "Warning: Column {:s} not contained in dataframe"
            msg = msg.format(str(element))
            print(msg)
            return False
    msg = "computing weights based on column(s): "
    msg += ", ".join(pColumnsToUse)
    print(msg)
    #compute the mask for samples to weight, order of bounds shouldn't matter
    #as long as min and max are defined
    weightingMask = False
    for column in pColumnsToUse:
        lowerMask = pDataframe[column] >= min(pBound1, pBound2)
        upperMask = pDataframe[column] <= max(pBound1, pBound2)
        weightingMask |= (lowerMask & upperMask)
    numberOfWeightedSamples = pDataframe[weightingMask].shape[0]
    numberOfUnweightedSamples = pDataframe.shape[0] - numberOfWeightedSamples
    if numberOfWeightedSamples == 0 or numberOfUnweightedSamples == 0:
        success = False
    else:
        print("number of non-emphasized samples: {:d}".format(numberOfUnweightedSamples))
        print("number of emphasized samples: {:d}".format(numberOfWeightedSamples))
        weightInt = np.uint32(np.round(pFactorFloat * numberOfUnweightedSamples / numberOfWeightedSamples))
        pDataframe.loc[weightingMask, 'weights'] = weightInt
        weightSum = pDataframe.loc[weightingMask, 'weights'].sum().astype('uint32')
        print("weight given: {:d}".format(weightInt))
        print("weight sum non-emphasized samples: {:d}".format(numberOfUnweightedSamples))
        print("weight sum emphasized samples: {:d}".format(weightSum))
        print("target factor weighted/unweighted: {:.3f}".format(pFactorFloat))
        print("actual factor weighted/unweighted: {:.3f}".format(weightSum/numberOfUnweightedSamples))
        success = weightInt != 1
    return success        

def tadWeighting(pInOutDataFrame, pParams, pTadDomainFile, pFactorFloat, pMaxDistance=500000):
    ### weight all samples within TADs shorter than pMaxDistance
    ### until (weightSum of weightedSamples) / (weightSum of unweighted samples) = pFactorFloat
    #if nothing works, all weights equally = 1.0
    pInOutDataFrame['weights'] = 1.0
    success = False
    #check inputs
    if not pTadDomainFile or pFactorFloat == 0: 
        return False
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
        return False
    if tadDomainDf.empty:
        msg = "Warning: tadDomainFile does not contain data for chromosome {:s}\n"
        msg += "Ignoring TADs"
        msg = msg.format(str(pParams['chrom']))
        print(msg)
        return False
    #compute weighting mask
    #there are only few thousand TADs in a chromosome, so iterating should be ok
    weightingMask = False
    for start, end in zip(tadDomainDf['chromStart'], tadDomainDf['chromEnd']):
        if end - start < pMaxDistance:
            startMask = pInOutDataFrame['first'] >= math.floor(start / int(pParams['resolution']))
            endMask = pInOutDataFrame['second'] <= math.floor(end / int(pParams['resolution']))
            weightingMask |= (startMask & endMask)
    numberOfWeightedSamples = pInOutDataFrame[weightingMask].shape[0]
    numberOfUnweightedSamples = pInOutDataFrame.shape[0] - numberOfWeightedSamples
    #give some feedback on the results
    if numberOfWeightedSamples == 0 or numberOfUnweightedSamples == 0:
        success = False
    else:
        weightInt = np.uint32(np.round(pFactorFloat * numberOfUnweightedSamples / numberOfWeightedSamples))
        pInOutDataFrame.loc[weightingMask, 'weights'] = weightInt
        weightSum = pInOutDataFrame.loc[weightingMask, 'weights'].sum().astype('uint32')
        print("weight given: {:d}".format(weightInt))
        print("weight sum non-emphasized samples: {:d}".format(numberOfUnweightedSamples))
        print("weight sum emphasized samples: {:d}".format(weightSum))
        print("target factor weighted/unweighted: {:.3f}".format(pFactorFloat))
        print("actual factor weighted/unweighted: {:.3f}".format(weightSum/numberOfUnweightedSamples))
        success = weightInt != 1
    return success
