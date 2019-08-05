from configurations import *

def createTag(resolution, cellLine,chrom, norm = None, eq = None,
               merge = None, window = None, ignore=None, windowSize = None,\
             model = None, loss=None, pc=None):
    tmp = 'R' + str(resolution) + "_" + cellLine 
    if norm:
        tmp += "_N"
    if eq:
        tmp += "_E"
    if ignore:
        tmp += "_A"
    if merge:
        tmp += "_M" +convertMethod(merge)
    if window:
        tmp += "_W" +convertMethod(window)
    if pc and pc != 6:
        tmp += "_PC" + str(pc)
    if model:
        tmp += "_" + model
    if loss:
        tmp += "_L" + loss


    return tmp + "_chr"+str(chrom)

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

