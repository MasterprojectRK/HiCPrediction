from configurations import *

def createTag(matrixFile,chrom, eq = None,
             window = None, ignore=None, windowSize = None,\
             model = None, loss=None, params = None):
    tmp = matrixFile.split("/")[-1].split(".")[0] 
    if params and params['normalize']:
        tmp += "_N"
    if eq:
        tmp += "_E"
    if ignore:
        tmp += "_A"
    if params and params['mergeOperation']:
        tmp += "_M" +convertMethod(params['mergeOperation'])
    if window:
        tmp += "_W" +convertMethod(window)
    if params and params['peakColumn'] and params['peakColumn']  != 6:
        tmp += "_PC" + str(pc)
    if model:
        tmp += "_" + model
    if loss:
        tmp += "_L" + loss

def createProteinTag(matrixFile,params):
    tmp = matrixFile.split("/")[-1].split(".")[0] 
    if params['normalize']:
        tmp += "_N"
    tmp += "_M" +params['mergeOperation']
    if params['peakColumn']  != 6:
        tmp += "_PC" + str(pc)
    return tmp


def convertMethod(method):
    if method == "avg":
        return "0"
    elif method == "max":
        return "1"
    elif method == "sum":
        return "2"

def chromsToName(s):
    return(chromListToString(chromStringToList(s)) )

def chromStringToList(s):
    chroms = []
    parts = s.split("_")
    for part in parts:
        elems = part.split("-")
        if len(elems) == 2:
            chroms.extend(range(int(elems[0]), int(elems[1])+1))
        elif len(elems) == 1:
            chroms.append(int(elems[0]))
        else:
            print("FAAAAAAAAAAAAAAAAAAAAAAAAAIIIIIIIIILLLLLLLLL")
    chroms = list(map(str, chroms))
    return chroms

def chromListToString(l):
    return ("_").join(l)

