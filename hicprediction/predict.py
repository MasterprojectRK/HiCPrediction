#!/usr/bin/env python3
import os
os.environ['NUMEXPR_MAX_THREADS'] = '16'
os.environ['NUMEXPR_NUM_THREADS'] = '8'
import hicprediction.configurations as conf
import click
from hicprediction.utilities import createPredictionTag, getResultFileColumnNames, checkExtension
import joblib
import pandas as pd
import numpy as np
from scipy import sparse
from scipy import ndimage
import sklearn.metrics as metrics
import sys
import math
import cooler
import sklearn.ensemble
from sklearn.preprocessing import MinMaxScaler

"""
Module responsible for the prediction of test set, their evaluation and the
conversion of prediction to HiC matrices
"""

@conf.predict_options
@click.command()
def executePredictionWrapper(modelfilepath, predictionsetpath,
                      predictionoutputdirectory, resultsfilepath, sigma, noconvertback):
    """
    Wrapper function for Cli
    """

    if not checkExtension(modelfilepath, '.z'):
        msg = "model file {0:s} does not have a .z file extension. Aborted"
        sys.exit(msg.format(modelfilepath))
    if not checkExtension(predictionsetpath, '.z'):
        msg = "prediction file {0:s} does not have a .z file extension. Aborted"
        sys.exit(msg.format(predictionsetpath))

    #load trained model and testSet (target for prediction)
    try:
        model, modelParams = joblib.load(modelfilepath)
        testSet, setParams = joblib.load(predictionsetpath)
    except Exception as e:
        print(e)
        msg = "Failed loading model and test set. Wrong format?"
        sys.exit(msg)
    #check if model and test set have been swapped
    if not isinstance(model, sklearn.ensemble.BaseEnsemble):
        msg = "Aborting. Input {:s} does not contain a Random Forest Regressor\n"
        if isinstance(model,pd.DataFrame):
            msg += "Maybe a dataset was entered instead of a trained model?"
        msg = msg.format(modelfilepath)
        sys.exit(msg)
    if not isinstance(testSet, pd.DataFrame):
        msg = "Aborting. Input {:s} is not a test dataset\n"
        if isinstance(testSet, sklearn.ensemble.BaseEnsemble):
            msg += "Maybe a trained model was entered instead of a dataset?"
        msg = msg.format(predictionsetpath)
        sys.exit(msg)

    executePrediction(model, modelParams, testSet, setParams,
                      predictionoutputdirectory, resultsfilepath, sigma, noconvertback                      )

def executePrediction(model,modelParams, testSet, setParams,
                      predictionoutputdirectory, resultsfilepath, sigma, noconvertback):
    """ 
    Main function
    calls prediction, evaluation and conversion methods and stores everything
    Attributes:
        model -- regression model
        modelParams -- parameters of model and set the model was trained with
        basefile-- path to basefile of test set
        predictionsetpath -- path to data set that is to be predicted
        predictionoutputdirectory -- path to store prediction
        resultsfilepath --  path to results file for evaluation storage
    """

    #check if the test set is a compound dataset (e.g. concatenated from diverse sets). 
    #this is not allowed for now
    if isinstance(setParams["chrom"], list) or isinstance(setParams["cellType"], list):
        msg = "The target dataset is a compound (concatenated) dataset with multiple chromosomes"
        msg += "or cell lines.\n" 
        msg += "Compound datasets cannot be predicted. Aborting"
        sys.exit(msg) 
        
    ### predict test dataset from model
    predictionDf, score = predict(model, testSet, modelParams, noconvertback)
        
    
    #prediction Tag for storing results
    predictionTag = createPredictionTag(modelParams, setParams)
    
    ### convert prediction back to matrix, if output path set
    if predictionoutputdirectory:
        predictionFilePath =  os.path.join(predictionoutputdirectory,predictionTag + ".cool")
        targetFilePath = os.path.join(predictionoutputdirectory,predictionTag + "_target.cool")
        #get target chromsize / max bin index, since the target matrix might be larger than the predicted one
        #because rows with zero protein entries may have been dropped at the front / end
        chromosome = setParams['chrom']
        resolutionInt = int(modelParams['resolution'])
        try:
            chromsize = setParams['chromSizes'][chromosome[3:]]
        except:
            msg = "No entry for original size of chromosome chr{:s} found.\n"
            msg += "Using size of predicted data, which may yield a smaller or larger predicted matrix"
            msg = msg.format(chromosome)
            print(msg)
            maxShapeIndx = max(int(predictionDf['first'].max()), int(predictionDf['second'].max())) + 1
            chromsize = maxShapeIndx * resolutionInt
        #set the correct matrix conversion function and convert
        if modelParams['method'] and modelParams['method'] == 'oneHot':
            convertToMatrix = predictionToMatrixOneHot
        elif modelParams['method'] and modelParams['method'] == 'multiColumn':
            convertToMatrix = predictionToMatrixMultiColumn
        else:
            msg = "Warning: model creation method unknown. Falling back to multiColumn"
            print(msg)
            convertToMatrix = predictionToMatrixMultiColumn
        
        #create a sparse matrix from the prediction dataframe
        predMatrix = convertToMatrix(predictionDf, modelParams['conversion'], chromsize, resolutionInt)
        #smoothen the predicted matrix with a gaussian filter, if sigma > 0.0
        if sigma > 0.0:
            predMatrix = smoothenMatrix(predMatrix, sigma)
            modelParams['smoothMatrix'] = sigma
        
        #create a sparse matrix for target, if available
        targetMatrix = None
        nanreadMask = testSet['reads'] == np.nan
        testSetHasTargetValues =  testSet[nanreadMask].empty
        if testSetHasTargetValues:
            rows = list(testSet['first'])
            columns = list(testSet['second'])
            matIndx = (rows,columns)
            maxShapeIndx = max( int(testSet['first'].max()), int(testSet['second'].max()) ) + 1
            targetData = testSet['reads']
            targetMatrix = sparse.csr_matrix((targetData, matIndx), shape=(maxShapeIndx, maxShapeIndx))

        #create and store final predicted matrix and target matrix, if available, in cooler format
        metadata = {"modelParams": modelParams, "targetParams": setParams}
        createCooler(predMatrix, chromosome, chromsize, resolutionInt, predictionFilePath, metadata)
        if targetMatrix != None:
            createCooler(targetMatrix, chromosome, chromsize, resolutionInt, targetFilePath, None)

    ### store evaluation metrics, if results path set
    if resultsfilepath:
        if score:
            saveResults(predictionTag, modelParams, setParams, predictionDf, testSet, score, resultsfilepath)
        else:
            msg = "Cannot evaluate prediction without target read values\n"
            msg += "Please provide a test set which contains target values\n"
            msg += "(or omit resultsfilepath)"
            print(msg)


def predict(model, pTestSet, pModelParams, pNoConvertBack):
    """
    Function to predict test set
    Attributes:
        model -- model to use
        testSet -- testSet to be predicted
        conversion -- conversion function used when training the model
    """
    #copy the test set, before invalidated rows and/or the columns not required for prediction are dropped
    predictionDf = pTestSet.copy(deep=True) 
    
    ### check if the test set contains reads, only then can we compute score later on
    nanreadMask = pTestSet['reads'] == np.nan
    testSetHasTargetValues =  pTestSet[nanreadMask].empty
    if not testSetHasTargetValues:
        predictionDf['reads'] = 0.0    
    
    ### remove invalidated rows
    validMask = predictionDf['valid'] == True
    predictionDf = predictionDf[validMask]
    if predictionDf.empty:
        msg = "Aborting. No valid samples to predict"
        raise SystemExit(msg)

    ### Eliminate NaNs - there should be none
    predictionDf.replace([np.inf, -np.inf], np.nan, inplace=True)
    if not predictionDf[predictionDf.isna().any(axis=1)].empty:
        msg = "Warning: There are {:d} rows in the testSet which contain NaN\n"
        msg = msg.format(predictionDf[predictionDf.isna().any(axis=1)].shape[0])
        msg += "The NaNs are in column(s) {:s}\n"
        msg = msg.format(", ".join(predictionDf[predictionDf.isna().any(axis=1)].columns))
        msg += "Replacing by zeros. Check input data!"
        print(msg)
        predictionDf.fillna(value=0, inplace=True)

    ### Hide Columns that are not needed for prediction
    dropList = ['first', 'second', 'chrom', 'reads', 'avgRead', 'valid']
    noDistance = 'noDistance' in pModelParams and pModelParams['noDistance'] == True
    noMiddle = 'noMiddle' in pModelParams and pModelParams['noMiddle'] == True
    noStartEnd = 'noStartEnd' in pModelParams and pModelParams['noStartEnd'] == True
    if noDistance:
        dropList.append('distance')
    if noMiddle:
        if pModelParams['method'] == 'oneHot':
            dropList.append('middleProt')
        elif pModelParams['method'] == 'multiColumn':
            numberOfProteins = int((predictionDf.shape[1] - 6) / 3)
            for protein in range(numberOfProteins):
                dropList.append(str(protein + numberOfProteins))
        else:
            raise NotImplementedError()
    if noStartEnd:
        if pModelParams['method'] == 'oneHot':
            dropList.append('startProt')
            dropList.append('endProt')
        elif pModelParams['method'] == 'multiColumn':
            numberOfProteins = int((predictionDf.shape[1] - 6) / 3)
            for protein in range(numberOfProteins):
                dropList.append(str(protein))
                dropList.append(str(protein + 2 * numberOfProteins))
        else:
            raise NotImplementedError()
    test_X = predictionDf.drop(columns=dropList, errors='ignore')

    ### convert reads to log reads
    predictionDf['standardLog'] = np.log(predictionDf['reads']+1)
    ### predict
    print("Valid prediction samples: {:d}".format(test_X.shape[0]))
    predReads = model.predict(test_X)
    if np.min(predReads) < 0:
        maxPred = np.max(predReads)
        np.clip(predReads, 0, None, out=predReads)
        msg = "Warning: Some predicted read counts were negative.\n"
        msg += "Clamping to range 0...{:.3f}".format(maxPred)
        print(msg)
    predictionDf['predReads'] = predReads
    ### clamp prediction output to normed input range, if desired
    if not pNoConvertBack \
        and pModelParams['normReadCount'] and pModelParams['normReadCount'] == True \
        and pModelParams['normReadCountValue'] and pModelParams ['normReadCountValue'] > 0:
        scaler = MinMaxScaler(feature_range=(0, pModelParams['normReadCountValue']), copy=False)
        predictionDf[['predReads']] = scaler.fit_transform(predictionDf[['predReads']])
        thresMask = predictionDf['predReads'] < pModelParams['normReadCountThreshold']
        predictionDf.loc[thresMask, 'predReads'] = 0.0
        msg = "normalized predicted values to range 0...{:.3f}, threshold {:.3f}"
        msg = msg.format(pModelParams['normReadCountValue'],pModelParams['normReadCountThreshold'])
        print(msg)
    #y_pred = np.absolute(y_pred)
    #test_y['predAbs'] = y_pred
    ### convert back if necessary
    if pModelParams['conversion'] == 'none':
        target = 'reads'
    elif pModelParams['conversion'] == 'standardLog':
        target = 'standardLog'
        predictionDf['predReads'] = np.exp(predictionDf['predReads']) - 1

    if testSetHasTargetValues:
        score = model.score(test_X,predictionDf[target])
    else:
        score = None
    return predictionDf, score

def predictionToMatrixOneHot(pPredictionDf, pConversion, pChromSize, pResolution):

    """
    Function to convert prediction to Hi-C matrix
    Attributes:
            pPredictionDf = Dataframe with predicted read counts in column 'predReads'
            pConversion = Name of conversion function
            pChromSize = (int) size of chromosome
            pResolution = (int) resolution of target HiC-Matrix in basepairs
    """
    ### store conversion function
    if pConversion == "standardLog":
        convert = lambda val: np.exp(val) - 1
    elif pConversion == "none":
        convert = lambda val: val
    else:
        msg = "unknown conversion type {:s}".format(str(pConversion))
        raise ValueError(msg)
    ### get individual predictions for the counts from each protein
    resList = []
    numberOfProteins = pPredictionDf.shape[1] - 13
    for protein in range(numberOfProteins):
        colName = 'prot_' + str(protein)
        mask = pPredictionDf[colName] == 1
        resDf = pd.DataFrame()
        resDf['first'] = pPredictionDf[mask]['first']
        resDf['second'] = pPredictionDf[mask]['second']
        ### convert back            
        predStr = 'pred_' + str(protein)
        resDf[predStr] = convert(pPredictionDf[mask]['predReads'])
        resDf.set_index(['first','second'],inplace=True)
        resList.append(resDf)

    #join the results on indices
    mergedPredictionDf = pd.DataFrame(columns=['first', 'second'])
    mergedPredictionDf.set_index(['first', 'second'], inplace=True)
    mergedPredictionDf = mergedPredictionDf.join(resList,how='outer')
    mergedPredictionDf.fillna(0.0, inplace=True)
    mergedPredictionDf['merged'] = mergedPredictionDf.mean(axis=1)
    #get the indices for the predicted counts
    mergedPredictionDf.reset_index(inplace=True)
    rows = list(mergedPredictionDf['first'])
    columns = list(mergedPredictionDf['second'])
    matIndx = (rows,columns)
    #get the predicted counts
    predData = list(mergedPredictionDf['merged'])
    #create predicted matrix
    maxShapeIndx = math.ceil(pChromSize / pResolution)
    predMatrix = sparse.csr_matrix((predData, matIndx), shape=(maxShapeIndx, maxShapeIndx))
    return predMatrix


def predictionToMatrixMultiColumn(pPredictionDf, pConversion, pChromSize, pResolution):

    """
    Function to convert prediction to Hi-C matrix
    Attributes:
            pPredictionDf = Dataframe with predicted read counts in column 'predReads'
            pConversion = Name of conversion function
            pChromSize = (int) size of chromosome
            pResolution = (int) resolution of target HiC-Matrix in basepairs
    """
    if pConversion == "standardLog":
        convert = lambda val: np.exp(val) - 1
    elif pConversion == "none":
        convert = lambda val: val
    else:
        msg = "unknown conversion type {:s}".format(str(pConversion))
        raise ValueError(msg)

    ### get rows and columns (indices) for re-building the HiC matrix
    rows = list(pPredictionDf['first'])
    columns = list(pPredictionDf['second'])
    matIndx = (rows,columns)
    ### convert back
    predData = list(convert(pPredictionDf['predReads']))
    ### create predicted matrix
    maxShapeIndx = math.ceil(pChromSize / pResolution)
    predMatrix = sparse.csr_matrix((predData, matIndx), shape=(maxShapeIndx, maxShapeIndx))
    return predMatrix


def createCooler(pSparseMatrix, pChromosome, pChromSize, pResolution, pOutfile, pMetadata):
    #get indices of upper triangular matrix
    triu_Indices = np.triu_indices(pSparseMatrix.shape[0])
    
    #create the bins for cooler
    bins = pd.DataFrame(columns=['chrom','start','end'])
    binStartList = list(range(0, pChromSize, int(pResolution)))
    binEndList = list(range(int(pResolution), pChromSize, int(pResolution)))
    binEndList.append(pChromSize)
    bins['start'] = binStartList
    bins['end'] = binEndList
    bins['chrom'] = str(pChromosome)

    #create the pixels for cooler
    pixels = pd.DataFrame(columns=['bin1_id','bin2_id','count'])
    pixels['bin1_id'] = triu_Indices[0]
    pixels['bin2_id'] = triu_Indices[1]
    readCounts = np.array(pSparseMatrix[triu_Indices])[0]
    pixels['count'] = np.float64(readCounts)
    pixels.sort_values(by=['bin1_id','bin2_id'],inplace=True)
    #write out the cooler
    cooler.create_cooler(pOutfile, bins=bins, pixels=pixels, dtypes={'count': np.float64}, metadata=pMetadata)


def smoothenMatrix(pSparseMatrix, pSigma):
        upper = sparse.triu(pSparseMatrix)
        lower = sparse.triu(pSparseMatrix, k=1).T
        fullPredMatrix = (upper+lower).todense().astype('float32')
        filteredPredMatrix = ndimage.gaussian_filter(fullPredMatrix,pSigma)
        predMatrix = sparse.triu(filteredPredMatrix, format='csr')
        return predMatrix


def getCorrelation(pData, pDistanceField, pTargetField, pPredictionField, pCorrMethod):
    """
    Helper method to calculate correlation
    """
    
    new = pData.groupby(pDistanceField, group_keys=False)[[pTargetField,
        pPredictionField]].corr(method=pCorrMethod)
    new = new.iloc[0::2,-1]
    #sometimes there is no variation in prediction / target per distance, then correlation is NaN
    #need to drop these, otherwise AUC will be NaN, too.
    new.dropna(inplace=True) 
    values = new.values
    indices = new.index.tolist()
    indices = list(map(lambda x: x[0], indices))
    indices = np.array(indices)
    div = pData[pDistanceField].max()
    indices = indices / div 
    return indices, values

def saveResults(pTag, pModelParams, pSetParams, pPredictionDf, pTargetDf, pScore, pResultsFilePath):
    """
    Function to calculate metrics and store them into a file
    """
    if not pResultsFilePath:
        return
    
    #prepare dataframe for prediction results
    if not checkExtension(pResultsFilePath, '.csv'):
        resultsfilename = os.path.splitext(pResultsFilePath)[0]
        pResultsFilePath = resultsfilename + ".csv"
        msg = "result file must have .csv file extension"
        msg += "renamed file to {0:s}"
        print(msg.format(pResultsFilePath))
    columns = getResultFileColumnNames(sorted(list(pTargetDf.distance.unique())))
    resultsDf = pd.DataFrame(columns=columns)
    resultsDf.set_index('Tag', inplace=True)
    
    targetColumnName = 'reads'
    predictedColumnName = 'predReads'
    # the test set may be larger than the prediction, 
    # if invalid values (no protein data) have been removed.
    # therefore join original test set (still contains all samples) with predicted one
    # and set non-existent predictions to zero
    jointPredictionTargetDf = pTargetDf.set_index(['first', 'second'])[[targetColumnName]].join( \
                                pPredictionDf.set_index(['first', 'second'])[[predictedColumnName, 'avgRead']], how='outer')
    jointPredictionTargetDf.reset_index(inplace=True)
    jointPredictionTargetDf['distance'] = jointPredictionTargetDf['second'] - jointPredictionTargetDf['first']
    jointPredictionTargetDf.fillna(0.0, inplace=True)                            
    y_pred = jointPredictionTargetDf[predictedColumnName]
    y_true = jointPredictionTargetDf[targetColumnName]
    ### calculate AUC for Pearson
    pearsonAucIndices, pearsonAucValues = getCorrelation(jointPredictionTargetDf,'distance', targetColumnName, predictedColumnName, 'pearson')
    pearsonAucScore = metrics.auc(pearsonAucIndices, pearsonAucValues)
    ### calculate AUC for Spearman
    spearmanAucIncides, spearmanAucValues = getCorrelation(jointPredictionTargetDf,'distance', targetColumnName, predictedColumnName, 'spearman')
    spearmanAucScore = metrics.auc(spearmanAucIncides, spearmanAucValues)
    corrScoreOPredicted_Pearson = jointPredictionTargetDf[[targetColumnName,predictedColumnName]].corr(method= \
                'pearson').iloc[0::2,-1].values[0]
    corrScoreOAverage_Pearson = jointPredictionTargetDf[[targetColumnName, 'avgRead']].corr(method= \
                'pearson').iloc[0::2,-1].values[0]
    corrScoreOPredicted_Spearman= jointPredictionTargetDf[[targetColumnName, predictedColumnName]].corr(method= \
                'spearman').iloc[0::2,-1].values[0]
    corrScoreOAverage_Spearman= jointPredictionTargetDf[[targetColumnName, 'avgRead']].corr(method= \
                'spearman').iloc[0::2,-1].values[0]
    print("PearsonAUC", pearsonAucScore)
    print("SpearmanAUC", spearmanAucScore)
    #model parameters cell type, chromosome, window operation and merge operation may be lists
    #so generate appropriate strings for storage
    modelCellTypeList = list( np.hstack([[], pModelParams['cellType']]) )
    modelChromList = list( np.hstack([[], pModelParams['chrom']]) )
    modelWindowOpList = list( np.hstack([[], pModelParams['windowOperation']]))
    modelMergeOpList = list( np.hstack([[], pModelParams['mergeOperation']]) )
    modelCellTypeStr = ", ".join(modelCellTypeList)
    modelChromStr = ", ".join(modelChromList)
    modelWindowOpStr = ", ".join(modelWindowOpList)
    modelMergeOpStr = ", ".join(modelMergeOpList)
    resultsDf.loc[pTag, 'Score'] = pScore 
    resultsDf.loc[pTag, 'R2'] = metrics.r2_score(y_true, y_pred)
    resultsDf.loc[pTag, 'MSE'] = metrics.mean_squared_error( y_true, y_pred)
    resultsDf.loc[pTag, 'MAE'] = metrics.mean_absolute_error( y_true, y_pred)
    resultsDf.loc[pTag, 'MSLE'] = metrics.mean_squared_log_error(y_true, y_pred)
    resultsDf.loc[pTag, 'AUC_OP_P'] = pearsonAucScore 
    resultsDf.loc[pTag, 'AUC_OP_S'] = spearmanAucScore
    resultsDf.loc[pTag, 'S_OP'] = corrScoreOPredicted_Spearman 
    resultsDf.loc[pTag, 'S_OA'] = corrScoreOAverage_Spearman
    resultsDf.loc[pTag, 'P_OP'] = corrScoreOPredicted_Pearson
    resultsDf.loc[pTag, 'P_OA'] = corrScoreOAverage_Pearson
    resultsDf.loc[pTag, 'Window'] = modelWindowOpStr
    resultsDf.loc[pTag, 'Merge'] = modelMergeOpStr,
    resultsDf.loc[pTag, 'normalize'] = pModelParams['normalize'] 
    resultsDf.loc[pTag, 'conversion'] = pModelParams['conversion'] 
    resultsDf.loc[pTag, 'Loss'] = 'MSE'
    resultsDf.loc[pTag, 'resolution'] = pModelParams['resolution']
    resultsDf.loc[pTag, 'modelChromosome'] = modelChromStr
    resultsDf.loc[pTag, 'modelCellType'] = modelCellTypeStr
    resultsDf.loc[pTag, 'predictionChromosome'] = pSetParams['chrom'] 
    resultsDf.loc[pTag, 'predictionCellType'] = pSetParams['cellType']
    for i, pearsonIndex in enumerate(pearsonAucIndices):
        columnName = int(round(pearsonIndex * jointPredictionTargetDf.distance.max()))
        resultsDf.loc[pTag, columnName] = pearsonAucValues[i]
    resultsDf = resultsDf.sort_values(by=['predictionCellType','predictionChromosome',
                            'modelCellType','modelChromosome', 'conversion',\
                            'Window','Merge', 'normalize'])
    resultsDf.to_csv(pResultsFilePath)
