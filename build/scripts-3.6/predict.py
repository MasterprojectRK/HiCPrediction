#!/home/ralf/miniconda3/envs/hicpred/bin/python
import os
os.environ['NUMEXPR_MAX_THREADS'] = '16'
os.environ['NUMEXPR_NUM_THREADS'] = '8'
import hicprediction.configurations as conf
import click
from hicprediction.tagCreator import createPredictionTag
import joblib
import pandas as pd
import numpy as np
import h5py
from scipy import sparse
from hicmatrix import HiCMatrix as hm
import sklearn.metrics as metrics

"""
Module responsible for the prediction of test set, their evaluation and the
conversion of prediction to HiC matrices
"""

@conf.predict_options
@click.command()
def executePredictionWrapper(modelfilepath, basefile, predictionsetpath,
                      predictionoutputdirectory, resultsfilepath):
    """
    Wrapper function for Cli
    """

    conf.checkExtension(modelfilepath, 'z')
    conf.checkExtension(predictionsetpath, 'z')
    model, modelParams = joblib.load(modelfilepath)
    testSet, setParams = joblib.load(predictionsetpath)
    executePrediction(model, modelParams, basefile, testSet, setParams,
                      predictionoutputdirectory, resultsfilepath)

def executePrediction(model,modelParams, basefile, testSet, setParams,
                      predictionoutputdirectory, resultsfilepath):
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
    ### check extensions
    conf.checkExtension(basefile, 'ph5')
    predictionTag = createPredictionTag(modelParams, setParams)
    if resultsfilepath:
        conf.checkExtension(resultsfilepath, 'csv')
        columns = [ 'Score', 'R2','MSE', 'MAE', 'MSLE',
                       'AUC_OP_S','AUC_OP_P', 'S_OP', 'S_OA', 'S_PA',
                       'P_OP','P_OA','P_PA',
                       'Window', 'Merge','equalize','normalize',
                       'ignoreCentromeres','conversion', 'Loss', 'Peak',
                       'resolution','modelChromosome', 'modelCellType',
                       'predictionChromosome', 'predictionCellType']
        columns.extend(list(range(modelParams['windowSize'])))
        columns.append('Tag')
        if os.path.isfile(resultsfilepath):
            df = pd.read_csv(resultsfilepath, index_col=0)
        else:
            df = pd.DataFrame(columns=columns)
            df = df.set_index('Tag')
        exists = predictionTag in df.index
    else:
        exists =False
    ### load model and set and predict
    if not exists:
        prediction, score = predict(model, testSet, modelParams['conversion'])
        if predictionoutputdirectory:
            predictionFilePath =  predictionoutputdirectory +"/"+ predictionTag + ".cool"
        ### call function to convert prediction to HiC matrix
            predictionToMatrix(prediction, basefile,modelParams['conversion'],\
                           setParams['chrom'], predictionFilePath)
        ### call function to store evaluation metrics
        if resultsfilepath:

            df = saveResults(predictionTag, df, modelParams, setParams, prediction,\
                        score, columns)
            df.to_csv(resultsfilepath)


def predict(model, testSet, conversion):
    """
    Function to predict test set
    Attributes:
        model -- model to use
        testSet -- testSet to be predicted
        conversion -- conversion function used when training the model
    """
    ### Eliminate NaNs
    testSet = testSet.fillna(value=0)
    ### Hide Columns that are not needed for prediction
    test_X = testSet[testSet.columns.difference(['first',
                                                 'second','chrom','reads',
                                                 'avgRead'])]
    test_y = testSet['chrom']
    test_y = test_y.to_frame()
    ### convert reads to log reads
    test_y['standardLog'] = np.log(testSet['reads']+1)
    ### predict
    y_pred = model.predict(test_X)
    test_y['pred'] = y_pred
    y_pred = np.absolute(y_pred)
    test_y['second'] = testSet['second']
    test_y['first'] = testSet['first']
    test_y['distance'] = testSet['distance']
    test_y['predAbs'] = y_pred
    ### convert back if necessary
    if conversion == 'none':
        target = 'reads'
        reads = y_pred
    elif conversion == 'standardLog':
        target = 'standardLog'
        reads = y_pred
        reads = np.exp(reads) - 1
    ### store into new dataframe
    test_y['reads'] = testSet['reads']
    test_y['avgRead'] = testSet['avgRead']
    test_y['predReads'] = reads
    score = model.score(test_X,test_y[target])
    test_y = test_y.set_index(['first','second'])
    return test_y, score

def predictionToMatrix(pred, baseFilePath,conversion, chromosome, predictionFilePath):

    """
    Function to convert prediction to Hi-C matrix
    Attributes:
            pred -- prediction dataframe
            baseFilePath --  base file
            conversion -- conversion technique that was used
            chromosome -- chromosome that wwas predicted
            predictionFilePath -- where to store the new matrix
    """
    with h5py.File(baseFilePath, 'r') as baseFile:
        ### store conversion function
        if conversion == "standardLog":
            convert = lambda val: np.exp(val) - 1
        elif conversion == "none":
            convert = lambda val: val
        ### get rows and columns
        rows = pred.index.codes[0]
        cols = pred.index.codes[1]
        data = convert(pred['pred'])
        ### convert back 
        ### create matrix with new values and overwrite original
        originalMatrix = hm.hiCMatrix(baseFile[chromosome].value)
        new = sparse.csr_matrix((data, (rows, cols)),\
                                shape=originalMatrix.matrix.shape)
        originalMatrix.setMatrix(new, originalMatrix.cut_intervals)
        originalMatrix.save(predictionFilePath)

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

def saveResults(tag, df, params, setParams, y, score, columns):
    """
    Function to calculate metrics and store them intoo a file
    """
    y_pred = y['predReads']
    y_true = y['reads']
    indicesOPP, valuesOPP= getCorrelation(y,'reads', 'predReads',
                                     params['resolution'], 'pearson')
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
    cols = [score, metrics.r2_score(y_true, y_pred),metrics.mean_squared_error( y_true, y_pred),
            metrics.mean_absolute_error( y_true, y_pred),
            metrics.mean_squared_log_error(y_true, y_pred),
            0, aucScoreOPP, corrScoreOP_S, corrScoreOA_S,
            0, corrScoreOP_P, corrScoreOA_P,
            0, params['windowOperation'],
            params['mergeOperation'],
            params['equalize'], params['normalize'], params['ignoreCentromeres'],
            params['conversion'], 'MSE', params['peakColumn'],
            params['resolution'],setParams['chrom'], setParams['cellType'],
            params['chrom'], params['cellType']]
    cols.extend(valuesOPP)
    df.loc[tag] = cols
    df = df.sort_values(by=['predictionCellType','predictionChromosome',
                            'modelCellType','modelChromosome', 'conversion',\
                            'Window','Merge', 'equalize', 'normalize'])
    return df
if __name__ == '__main__':
    executePredictionWrapper()
