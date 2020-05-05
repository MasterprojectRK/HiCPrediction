import numpy as np

def createSetTag(params):
    #compound sets may comprise multiple cell types, window operations and chromosomes
    cellTypeList = list( np.hstack([[], params['cellType']]) )
    windowOpList = list( np.hstack([[], params['windowOperation']]) )
    chromList = list( np.hstack([[], params['chrom']]) )
    cellTypeStr = "_".join(cellTypeList)
    windowOpStr = "_".join(windowOpList)
    chromStr = "_".join(chromList)
    tmp = cellTypeStr + '_' + str(params['resolution'])
    tmp +='_'+ createProteinTag(params)
    tmp += '_W' + windowOpStr
    tmp += str(params['windowSize'])
    tmp += '_B' + chromStr
    return tmp

def createProteinTag(params):
    #compound sets may comprise multiple merge operations
    mergeOpList = list( np.hstack([[], params['mergeOperation']]) )
    mergeOpStr = "_".join(mergeOpList)
    tmp = ''
    tmp += 'M' + mergeOpStr
    return tmp


def createModelTag(params):
    tmp = createSetTag(params)
    tmp +=  '_C' + params['conversion']
    if params['noDistance']:
        tmp += '_noDist'
    if params['noMiddle']:
        tmp += '_noMiddle'
    if params['noStartEnd']:
        tmp += '_noStartEnd'
    return tmp

def createPredictionTag(params, setParams):
    tmp = 'Model_' + createModelTag(params)
    tmp += '_PredictionOn_' + setParams['cellType']
    tmp += "_" + setParams['chrom']
    return tmp


def initParamDict():
    paramNamesList = [
        'resolution',
        'cellType',
        'chromSizes',
        'smoothProt',
        'chrom',
        'windowOperation',
        'mergeOperation',
        'matrixCorrection',
        'normalize',
        'conversion',
        'normSignalValue',
        'normSignalThreshold',
        'normReadCount',
        'normReadCountValue',
        'normReadCountThreshold',
        'windowSize',
        'removeEmpty',
        'method',
        'noDistance',
        'noMiddle',
        'noStartEnd',
        'noDiagonal',
        'useExtraTrees',
        'learningParams',
        'smoothMatrix',
        'proteinFileNames', 
    ]
    paramDict = {paramName: None for paramName in paramNamesList}
    return paramDict

def getResultFileColumnNames(distList):
    columns = ['Score', 
                    'R2',
                    'MSE', 
                    'MAE', 
                    'MSLE',
                    'AUC_OP_S',
                    'AUC_OP_P', 
                    'S_OP', 
                    'S_OA', 
                    'P_OP',
                    'P_OA',
                    'Window', 
                    'Merge',
                    'normalize',
                    'conversion', 
                    'Loss', 
                    'resolution',
                    'modelChromosome', 
                    'modelCellType',
                    'predictionChromosome', 
                    'predictionCellType']    
    columns.extend(distList)
    columns.append('Tag')
    return columns

def normalizeDataFrameColumn(pDataFrame, pColumnName, pMaxValue, pThreshold):
    #inplace zero-to-pMaxValue normalization of a column in a dataframe
    #values below pThreshold will be set to zero, if pThreshold < pMaxValue
    if not pColumnName in pDataFrame.columns:
        return
    
    columnMax = pDataFrame[pColumnName].max()
    columnMin = pDataFrame[pColumnName].min()
    if columnMin == columnMax:
        msg = "no variance in column {:s} of dataframe".format(str(pColumnName))
        pDataFrame[pColumnName] = 0.0
        raise Warning(msg)
    else: #zero-to-one normalization first, then multiply with pMaxValue
        diff = columnMax - columnMin
        pDataFrame[pColumnName] = ((pDataFrame[pColumnName] - columnMin) / diff).astype('float32')
        pDataFrame[pColumnName] *= pMaxValue
    if pThreshold < pMaxValue:
        setToZeroMask = pDataFrame[pColumnName] < pThreshold
        pDataFrame.loc[setToZeroMask, pColumnName] = 0.0 #no-op if all values > threshold