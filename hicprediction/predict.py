#!/usr/bin/env python3
import os
os.environ['NUMEXPR_MAX_THREADS'] = '16'
os.environ['NUMEXPR_NUM_THREADS'] = '8'
import hicprediction.configurations as conf
import click
from hicprediction.tagCreator import createPredictionTag
import joblib
import pandas as pd
import numpy as np
from scipy import sparse
from scipy import ndimage
import sklearn.metrics as metrics
import sys
import math
import cooler

"""
Module responsible for the prediction of test set, their evaluation and the
conversion of prediction to HiC matrices
"""

@conf.predict_options
@click.command()
def executePredictionWrapper(modelfilepath, predictionsetpath,
                      predictionoutputdirectory, resultsfilepath, sigma):
    """
    Wrapper function for Cli
    """

    if not conf.checkExtension(modelfilepath, '.z'):
        msg = "model file {0:s} does not have a .z file extension. Aborted"
        sys.exit(msg.format(modelfilepath))
    if not conf.checkExtension(predictionsetpath, '.z'):
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

    executePrediction(model, modelParams, testSet, setParams,
                      predictionoutputdirectory, resultsfilepath, sigma)

def executePrediction(model,modelParams, testSet, setParams,
                      predictionoutputdirectory, resultsfilepath, sigma):
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
    
    #prepare dataframe for prediction results
    df = None
    if resultsfilepath:
        if not conf.checkExtension(resultsfilepath, '.csv'):
            resultsfilename = os.path.splitext(resultsfilepath)[0]
            resultsfilepath = resultsfilename + ".csv"
            msg = "result file must have .csv file extension"
            msg += "renamed file to {0:s}"
            print(msg.format(resultsfilepath))
        columns = ['Score', 
                    'R2',
                    'MSE', 
                    'MAE', 
                    'MSLE',
                    'AUC_OP_S',
                    'AUC_OP_P', 
                    'S_OP', 
                    'S_OA', 
                    'S_PA',
                    'P_OP',
                    'P_OA',
                    'P_PA',
                    'Window', 
                    'Merge',
                    'normalize',
                    'ignoreCentromeres',
                    'conversion', 
                    'Loss', 
                    'resolution',
                    'modelChromosome', 
                    'modelCellType',
                    'predictionChromosome', 
                    'predictionCellType']
        dists = sorted(list(testSet.distance.unique()))
        columns.extend(dists)
        columns.append('Tag')
        df = pd.DataFrame(columns=columns)
        df = df.set_index('Tag')
    
    ### predict test dataset from model
    predictionDf, score = predict(model, testSet, modelParams)
    
    #prediction Tag for storing results
    predictionTag = createPredictionTag(modelParams, setParams)
    
    ### convert prediction back to matrix, if output path set
    if predictionoutputdirectory:
        predictionFilePath =  os.path.join(predictionoutputdirectory,predictionTag + ".cool")
        #get target chromsize / max bin index, since the target matrix might be larger than the predicted one
        #because rows with zero protein entries may have been dropped at the front / end
        chromosome = setParams['chrom']
        resolutionInt = int(modelParams['resolution'])
        try:
            chromsize = modelParams['chromSizes'][chromosome[3:]]
        except:
            msg = "No entry for original size of chromosome chr{:s} found.\n"
            msg += "Using size of predicted data, which may yield a smaller or larger predicted matrix"
            msg = msg.format(chromosome)
            print(msg)
            maxShapeIndx = max(int(predictionDf['first'].max()), int(predictionDf['second'].max()))
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
        #create and store final predicted matrix in cooler format
        createCooler(predMatrix, chromosome, chromsize, resolutionInt, predictionFilePath)

    ### store evaluation metrics, if results path set
    if resultsfilepath:
        if score:
            df = saveResults(predictionTag, df, modelParams, setParams, predictionDf, score, columns)
            df.to_csv(resultsfilepath)
        else:
            msg = "Cannot evaluate prediction without target read values\n"
            msg += "Please provide a test set which contains target values\n"
            msg += "(or omit resultsfilepath)"
            print(msg)


def predict(model, testSet, pModelParams):
    """
    Function to predict test set
    Attributes:
        model -- model to use
        testSet -- testSet to be predicted
        conversion -- conversion function used when training the model
    """
    ### check if the test set contains reads, only then can we compute score later on
    nanreadMask = testSet['reads'] == np.nan
    testSetHasTargetValues =  testSet[nanreadMask].empty    
    
    ### Eliminate NaNs - there should be none
    testSet.fillna(value=0, inplace=True)

    ### Hide Columns that are not needed for prediction
    dropList = ['first', 'second', 'chrom', 'reads', 'avgRead']
    noDistance = 'noDistance' in pModelParams and pModelParams['noDistance'] == True
    noMiddle = 'noMiddle' in pModelParams and pModelParams['noMiddle'] == True
    noStartEnd = 'noStartEnd' in pModelParams and pModelParams['noStartEnd'] == True
    if noDistance:
        dropList.append('distance')
    if noMiddle:
        if pModelParams['method'] == 'oneHot':
            dropList.append('middleProt')
        elif pModelParams['method'] == 'multiColumn':
            numberOfProteins = int((testSet.shape[1] - 6) / 3)
            for protein in range(numberOfProteins):
                dropList.append(str(protein + numberOfProteins))
        else:
            raise NotImplementedError()
    if noStartEnd:
        if pModelParams['method'] == 'oneHot':
            dropList.append('startProt')
            dropList.append('endProt')
        elif pModelParams['method'] == 'multiColumn':
            numberOfProteins = int((testSet.shape[1] - 6) / 3)
            for protein in range(numberOfProteins):
                dropList.append(str(protein))
                dropList.append(str(protein + 2 * numberOfProteins))
        else:
            raise NotImplementedError()
    test_X = testSet[testSet.columns.difference(dropList)] #also works if one of the columns to drop is not present
    test_y = testSet.copy(deep=True)
    ### convert reads to log reads
    test_y['standardLog'] = np.log(testSet['reads']+1)
    ### predict
    y_pred = model.predict(test_X)
    test_y['pred'] = y_pred
    y_pred = np.absolute(y_pred)
    test_y['predAbs'] = y_pred
    ### convert back if necessary
    if pModelParams['conversion'] == 'none':
        target = 'reads'
        reads = y_pred
    elif pModelParams['conversion'] == 'standardLog':
        target = 'standardLog'
        reads = np.exp(y_pred) - 1
    ### store into new dataframe
    test_y['predReads'] = reads
    if testSetHasTargetValues:
        score = model.score(test_X,test_y[target])
    else:
        score = None
    return test_y, score

def predictionToMatrixOneHot(pPredictionDf, pConversion, pChromSize, pResolution):

    """
    Function to convert prediction to Hi-C matrix
    Attributes:
            pPredictionDf = Dataframe with predicted read counts in column 'pred'
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
        resDf[predStr] = convert(pPredictionDf[mask]['pred'])
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
    data = list(mergedPredictionDf['merged'])
        
    #create predicted matrix
    maxShapeIndx = math.ceil(pChromSize / pResolution)
    predMatrix = sparse.csr_matrix((data, matIndx), shape=(maxShapeIndx, maxShapeIndx))
    return predMatrix


def predictionToMatrixMultiColumn(pPredictionDf, pConversion, pChromSize, pResolution):

    """
    Function to convert prediction to Hi-C matrix
    Attributes:
            pPredictionDf = Dataframe with predicted read counts in column 'pred'
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
    data = list(convert(pPredictionDf['pred']))
    ### create predicted matrix
    maxShapeIndx = math.ceil(pChromSize / pResolution)
    predMatrix = sparse.csr_matrix((data, matIndx), shape=(maxShapeIndx, maxShapeIndx))
    return predMatrix


def createCooler(pSparseMatrix, pChromosome, pChromSize, pResolution, pOutfile):
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
    pixels['count'] = np.float32(readCounts)
    pixels.sort_values(by=['bin1_id','bin2_id'],inplace=True)

    #write out the cooler
    cooler.create_cooler(pOutfile, bins=bins, pixels=pixels)


def smoothenMatrix(pSparseMatrix, pSigma):
        upper = sparse.triu(pSparseMatrix)
        lower = sparse.triu(pSparseMatrix, k=1).T
        fullPredMatrix = (upper+lower).todense().astype('float32')
        filteredPredMatrix = ndimage.gaussian_filter(fullPredMatrix,pSigma)
        predMatrix = sparse.triu(filteredPredMatrix)
        return predMatrix


def getCorrelation(data, field1, field2,  resolution, method):
    """
    Helper method to calculate correlation
    """

    new = data.groupby('distance', group_keys=False)[[field1,
        field2]].corr(method=method)
    new = new.iloc[0::2,-1]
    values = new.values
    indices = new.index.tolist()
    indices = list(map(lambda x: x[0], indices))
    indices = np.array(indices)
    div = float(len(indices))
    indices = indices / div 
    return indices, values

def saveResults(pTag, df, pModelParams, pSetParams, y, pScore, pColumns):
    """
    Function to calculate metrics and store them into a file
    """
    y_pred = y['predReads']
    y_true = y['reads']
    indicesOPP, valuesOPP = getCorrelation(y,'reads', 'predReads',
                                     pModelParams['resolution'], 'pearson')
    ### calculate AUC
    aucScoreOPP = metrics.auc(indicesOPP, valuesOPP)
    corrScoreOP_P = y[['reads','predReads']].corr(method= \
                'pearson').iloc[0::2,-1].values[0]
    corrScoreOA_P = y[['reads', 'avgRead']].corr(method= \
                'pearson').iloc[0::2,-1].values[0]
    corrScoreOP_S= y[['reads','predReads']].corr(method= \
                'spearman').iloc[0::2,-1].values[0]
    corrScoreOA_S= y[['reads', 'avgRead']].corr(method= \
                'spearman').iloc[0::2,-1].values[0]
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
    cols = [pScore, 
            metrics.r2_score(y_true, y_pred),
            metrics.mean_squared_error( y_true, y_pred),
            metrics.mean_absolute_error( y_true, y_pred),
            metrics.mean_squared_log_error(y_true, y_pred),
            0, 
            aucScoreOPP, 
            corrScoreOP_S, 
            corrScoreOA_S,
            0, 
            corrScoreOP_P, 
            corrScoreOA_P,
            0, 
            modelWindowOpStr,
            modelMergeOpStr,
            pModelParams['normalize'], 
            pModelParams['ignoreCentromeres'],
            pModelParams['conversion'], 
            'MSE', 
            pModelParams['resolution'],
            pSetParams['chrom'], 
            pSetParams['cellType'],
            modelChromStr, 
            modelCellTypeStr]
    cols.extend(valuesOPP)
    df.loc[pTag] = cols
    df = df.sort_values(by=['predictionCellType','predictionChromosome',
                            'modelCellType','modelChromosome', 'conversion',\
                            'Window','Merge', 'normalize'])
    return df
if __name__ == '__main__':
    executePredictionWrapper()
