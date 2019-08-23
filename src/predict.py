#!/usr/bin/env python3

from src.configurations import *
from src.tagCreator import createPredictionTag

"""
Module responsible for the prediction of test set, their evaluation and the
conversion of prediction to HiC matrices
"""

@predict_options
@click.command()
def executePrediction(modelfilepath, basefilepath, predictionsetpath, 
                      predictionoutputdirectory, resultsfilepath):
    """ 
    Main function
    calls prediction, evaluation and conversion methods and stores everything
    Attributes:
        modelfilepath -- path to model that is to be used
        basefilepath -- path to basefile of test set
        predictionsetpath -- path to data set that is to be predicted
        predictionoutputdirectory -- path to store prediction
        resultsfilepath --  path to results file for evaluation storage
    """
    ### check extensions
    checkExtension(modelfilepath, 'z')
    checkExtension(predictionsetpath, 'z')
    checkExtension(basefilepath, 'ph5')
    checkExtension(resultsfilepath, 'csv')
    ### load model and set and predict
    model, modelParams = joblib.load(modelfilepath)
    testSet, setParams = joblib.load(predictionsetpath)
    prediction, score = predict(model, testSet, modelParams['conversion'])
    predictionTag = createPredictionTag(modelParams, setParams)
    predictionFilePath =  predictionoutputdirectory + predictionTag + ".cool"
    ### call function to convert prediction to HiC matrix
    predictionToMatrix(prediction, basefilepath,modelParams['conversion'],\
                       setParams['chrom'], predictionFilePath)
    ### call function to store evaluation metrics
    if resultsfilepath:
        saveResults(resultsfilepath, modelParams, setParams, prediction, score)

def predict(model, testSet, conversion):
    """
    Function to predict test set
    Attributes:
        model -- model to use
        testSet -- testSet to be predicted
        conversion -- conversion function used when training the model
    """
    testSet = testSet.fillna(value=0)
    test_X = testSet[testSet.columns.difference(['first', 'second','chrom','reads'])]
    test_y = testSet['chrom']
    test_y = test_y.to_frame()
    test_y['standardLog'] = np.log(testSet['reads']+1)
    y_pred = model.predict(test_X)
    test_y['pred'] = y_pred
    y_pred = np.absolute(y_pred)
    test_y['second'] = testSet['second']
    test_y['first'] = testSet['first']
    test_y['distance'] = testSet['distance']
    test_y['predAbs'] = y_pred
    # if args.conversion == "norm":
        # target = 'normTarget'
        # reads = y_pred * maxV
    # elif args.conversion == "log":
        # target = 'logTarget'
        # reads = y_pred * np.log(maxV)
        # reads = np.exp(reads) - 1
    if conversion == 'none':
        target = 'reads'
        reads = y_pred
    elif conversion == 'standardLog':
        target = 'standardLog'
        reads = y_pred
        reads = np.exp(reads) - 1
    test_y['reads'] = testSet['reads']
    test_y['avgRead'] = testSet['avgRead']
    test_y['predReads'] = reads
    score = model.score(test_X,test_y[target])
    test_y = test_y.set_index(['first','second'])
    return test_y, score

def predictionToMatrix(pred, baseFilePath,conversion, chromosome, predictionFilePath):
    with h5py.File(baseFilePath, 'r') as baseFile:
        if conversion == "standardLog":
            convert = lambda val: np.exp(val) - 1
        elif conversion == "none":
            convert = lambda val: val
        rows = pred.index.codes[0]
        cols = pred.index.codes[1]
        data = convert(pred['pred'])
        originalMatrix = hm.hiCMatrix(baseFile[chromosome].value)
        new = sparse.csr_matrix((data, (rows, cols)),\
                                shape=originalMatrix.matrix.shape)
        originalMatrix.setMatrix(new, originalMatrix.cut_intervals)
        originalMatrix.save(predictionFilePath)

def getPearson(data, field1, field2,  resolution):
    new = data.groupby('distance', group_keys=False)[[field1,
        field2]].corr(method='spearman')
    new = new.iloc[0::2,-1]
    values = new.values
    indices = new.index.tolist()
    indices = list(map(lambda x: x[0], indices))
    indices = np.array(indices)
    div = float(len(indices))
    indices = indices / div 
    return indices, values

def saveResults(resultsfilepath, params, setParams, y, score):
    y_pred = y['predReads']
    y_true = y['reads']
    indicesOP, valuesOP = getPearson(y,'reads', 'predReads', params['resolution'])
    aucScoreOPS = auc(indicesOP, valuesOP)
    aucScoreOP = y[['reads','predReads']].corr(method= \
                'spearman').iloc[0::2,-1].values[0]
    aucScoreOA = y[['reads', 'avgRead']].corr(method= \
                'spearman').iloc[0::2,-1].values[0]
    aucScorePA = y[['predReads', 'avgRead']].corr(method=\
                'spearman').iloc[0::2,-1].values[0]
    print(aucScoreOP, aucScorePA, aucScoreOA)
    columns = ['Score', 'R2','MSE', 'MAE', 'MSLE',
                   'AUC_OP_S',
                   'P_OP','P_OA','P_PA',
                   'Window', 'Merge','equalize','normalize',
                   'ignoreCentromeres','conversion', 'Loss', 'Peak',
                   'resolution','modelChromosome', 'modelCellType',
                   'predictionChromosome', 'predictionCellType']
    columns.extend(list(range(params['windowSize'])))
    if os.path.isfile(resultsfilepath):
        df = pd.read_csv(resultsfilepath)
    else:
        df = pd.DataFrame(columns=columns)
    cols = [score, r2_score(y_pred,y_true),mean_squared_error(y_pred, y_true),
            mean_absolute_error(y_pred, y_true), mean_squared_log_error(y_pred, y_true),
            aucScoreOPS, aucScoreOP, aucScoreOA, aucScorePA, params['windowOperation'],
            params['mergeOperation'],
            params['equalize'], params['normalize'], params['ignoreCentromeres'],
            params['conversion'], params['lossfunction'], params['peakColumn'],
            params['resolution'], params['chrom'], params['cellType'],
            setParams['chrom'], setParams['cellType']]
    cols.extend(valuesOP)
    s = pd.Series(cols, index=columns)
    df = df.append(s, ignore_index=True)
    df = df.sort_values(by=['predictionCellType','predictionChromosome',
                            'modelCellType','modelChromosome', 'conversion',\
                            'Window','Merge', 'equalize', 'normalize'])
    df.drop(list(df.filter(regex = 'Unnamed*')), axis = 1, inplace = True)
    print(df.iloc[:,:22].head())
    df.to_csv(resultsfilepath)

if __name__ == '__main__':
    executePrediction()
