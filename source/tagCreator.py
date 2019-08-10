from configurations import *

def createSetTag(proteinTag,chromTag, window,\
                 ignore=None, eq = None,windowSize = None):
    tmp = proteinTag 
    tmp += "_W" + window +str(windowSize)
    if eq:
        tmp += "_E"
    if ignore:
        tmp += "_A"
    return tmp + chromTag

def createProteinTag(params):
    tmp = "M" +params['mergeOperation']
    if params['normalize']:
        tmp += "_N"
    if params['peakColumn']  != 6:
        tmp += "_PC" + str(pc)
    return tmp


def createModelTag(setTag, conversion, lossfunction):
    return setTag + "_C" + conversion + "_L" + lossfunction
