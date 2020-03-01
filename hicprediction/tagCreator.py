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
