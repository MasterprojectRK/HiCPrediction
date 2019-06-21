from hicmatrix import HiCMatrix as hm
import torch
from hicexplorer import hicPlotMatrix as hicPlot
import numpy as np
from scipy import sparse
import os
import logging as log
import sys
from collections import deque
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import argparse
import pickle
import cooler
from copy import copy, deepcopy
from scipy import signal
from scipy import misc
import pybedtools
import math
import time
import pandas as pd

log.basicConfig(level=log.DEBUG)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=3, suppress=True)
# global constants
BIN_D = "20B/"
DATA_D = "Data2e/"
CHROM_D = DATA_D +BIN_D+ "Chroms/"
ARM_D = DATA_D +BIN_D+ "Arms/"
SET_D = DATA_D + BIN_D +"Sets/"
PRED_D = DATA_D +BIN_D+ "Predictions/"
MODEL_D  = DATA_D + BIN_D+"Models/"
CHUNK_D = DATA_D +BIN_D +"Chunks/"
IMAGE_D = DATA_D +  "Images/"
ORIG_D = DATA_D +  "BaseData/Orig/"
PROTEIN_D = DATA_D + BIN_D+"Proteins/"
PROTEINORIG_D = DATA_D +"BaseData/ProteinOrig/"
allProteins = pd.DataFrame()
PREDTYPE = "pred.p"
COOL = ".cool"
def chrom_filter(feature, c):
        return feature.chrom == c

def peak_filter(feature, s, e):
        peak = feature.start + int(feature[9])
        return s <= peak and e >= peak


def predictionToMatrix(args, pred=None):
    if pred is None:
        pred = pickle.load(open(tagCreator(args, "pred"), "rb" ) )
    ma = hm.hiCMatrix(ARM_D +args.chrom+".cool")
    mat = ma.matrix.todense()
    factor =  np.max(mat)
    new = np.zeros(mat.shape)
    print(new.shape)
    cuts = ma.cut_intervals
    l = len(cuts)
    for j, cut in enumerate(cuts):
        maxV = min(l - j - 1,int(args.reach)+1)
        print(str(j+1)+"/"+str(len(cuts)),end='')
        print('\r', end='') # use '\r' to go back
        sys.stdout.flush()

        for i in range(2,maxV):
            val = pred.loc[(cut[1], cuts[j+i][1])]['pred']
            if args.conversion == "norm":
                val= val  * factor
            elif args.conversion == "log":
                val= val  * np.log(factor)
                val = np.exp(val) - 1
            new[j][j+i] = val
    new = sparse.csr_matrix(new)
    ma.setMatrix(new, ma.cut_intervals)
    ma.save(tagCreator(args, "pred"))

def divideIntoArms(args):
    ma = hm.hiCMatrix(CHROM_D +args.chrom+".cool")

    f=open("Data2e/BaseData/centromeres.txt", "r")
    fl =f.readlines()
    elems = None
    for x in fl:
        elems = x.split("\t")
        if elems[1] == "chr"+args.chrom:
            print(elems)
            break
    start = int(elems[2])
    end = int(elems[3])
    cuts = ma.cut_intervals
    i = 0
    cuts1 = []
    cuts2 = []
    print(cuts[4510:4530])
    firstIndex = 0
    for cut in cuts:
        
        if cut[2] < start:
            cuts1.append(cut)
            lastIndex = i + 1
        elif cut[1] > end:
            cuts2.append(cut)
        else:
            firstIndex = i + 1
        i += 1
    if firstIndex == 0:
        firstIndex = lastIndex
    print(len(cuts))
    print(len(cuts1))
    print(len(cuts2))

    m1 = ma.matrix.todense()
    m2 = ma.matrix.todense()
    m1 = m1[:lastIndex,:lastIndex]
    new = sparse.csr_matrix(m1)
    ma.setMatrix(new, cuts1)
    ma.save(ARM_D + args.chrom + "_A.cool")
    
    m2 = m2[firstIndex:,firstIndex:]
    new = sparse.csr_matrix(m2)
    ma.setMatrix(new, cuts2)
    ma.save(ARM_D + args.chrom + "_B.cool")

def createAllArms(args):
    # for i in range(1,13):
        # args.chrom = str(i)
        # divideIntoArms(args)
    for i in range(16,22):
        args.chrom = str(i)
        divideIntoArms(args)


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

def chromsToName(s):
    return(chromListToString(chromStringToList(s)) )


def tagCreator(args, mode):
    if args.equalizeProteins:
        ep = "_E"
    else:
        ep = ""
    if args.normalizeProteins:
        nop = "_N"
    else:
        nop = ""
    
    wmep = "_"+args.windowOperation +"_M"+args.mergeOperation+ep +nop

    cowmep =  "_"+args.conversion +  wmep
    csa = chromsToName(args.chroms)+"_"+args.arms
    csam = csa + "_"+args.model

    if mode == "set":
        return SET_D + args.chrom + wmep +".p"

    elif mode == "model":
        return MODEL_D + csam + cowmep+".p"

    elif mode == "protein":
        return PROTEIN_D + args.chrom+ "_M"+args.mergeOperation+nop+".p"
    elif mode == "pred":
        return PRED_D + args.chrom + "_P"+ csam + cowmep +".cool"

    elif mode == "setC":
        return SET_D+csa+ wmep +".p"

    elif mode == "image":
        return IMAGE_D + args.chrom +"_R" + args.region+"_P"+csam + cowmep +".png"

def createForestDataset(args):
    allProteins = pickle.load(open(tagCreator(args, "protein"), "rb" )).values.tolist()
    colNr = np.shape(allProteins)[1]
    rows = np.shape(allProteins)[0]
    ma = hm.hiCMatrix(ARM_D+args.chrom+".cool").matrix
    reads = ma.todense()
    logs = deepcopy(reads)
    norms = deepcopy(reads)
    
    maxValue = 25790 
    logs = np.log(logs+1)
    logs /= np.log(maxValue)
    norms /= maxValue
    print(ma.shape)
    print(rows,colNr)
    cols = ['first', 'second','chrom'] + list(range(3*
         (colNr-1)))+['distance','reads','logTarget','normTarget']
    window = []
    for j in range(rows):
        maxReach = min(int(args.reach)+1,rows-j)
        ownProteins = allProteins[j][1:]
        firstStart = allProteins[j][0]
        for  i in range(2,maxReach):
            secondStart = allProteins[j+i][0]
            distance = secondStart - firstStart
            secondProteins = allProteins[j+i][1:]
            if j+1 >= j+i:
                middleProteins = np.zeros(colNr -1)
            else:
                middleProteins = allProteins[j+1:j+i]
                if args.windowOperation == 'avg':
                    middleProteins = np.mean(middleProteins, axis=0)[1:]
                elif args.windowOperation == 'sum':
                    middleProteins = np.sum(middleProteins, axis=0)[1:]
                elif args.windowOperation == 'max':
                    middleProteins = np.max(middleProteins, axis=0)[1:]

            frame = [firstStart, secondStart,args.chrom]
            if args.equalizeProteins:
                for l in range(len(ownProteins)):
                    if ownProteins[l] == 0:
                        secondProteins[l] = 0
                    elif secondProteins[l] == 0:
                        ownProteins[l] = 0
            frame.extend(ownProteins)
            frame.extend(middleProteins)
            frame.extend(secondProteins)
            frame.append(distance)
            valR = reads[j,j+i]
            valL = logs[j,j+i]
            valN = norms[j,j+i]
            frame.append(valR)
            frame.append(valL)
            frame.append(valN)
            window.append(frame)
    data = pd.DataFrame(window,columns =cols)
    print(data.shape)
    pickle.dump(data, open(tagCreator(args, "set"), "wb" ) )



def createAllWindows(args):
    for f in os.listdir(ARM_D):
        args.chrom = f.split(".")[0]
        createForestDataset(args)

def loadAllProteins(args):
    for f in os.listdir(ARM_D):
        args.chrom = f.split(".")[0]
        loadProtein(args)

def score(feature):
    return feature.score

def loadProtein(args):
    ma = hm.hiCMatrix(ARM_D+args.chrom+".cool")
    fullChrom = args.chrom.split("_")[0]
    cuts = ma.cut_intervals
    i = 0
    allProteins = []
    for cut in cuts:
        allProteins.append(np.zeros(15))
        allProteins[i][0] = cut[1]
        i += 1
    i = 0
    for f in os.listdir(PROTEINORIG_D):
        print(f)
        path  = PROTEINORIG_D+f
        a = pybedtools.BedTool(path)
        b = a.to_dataframe()
        c = b.iloc[:,6]
        maxV = max(c)
        if maxV == 0:
            maxV = 1
        a = a.filter(chrom_filter, c='chr'+fullChrom)
        a = a.sort()
        j = 0
        for cut in cuts:    
            print(str(j+1)+"/"+str(len(cuts)),end='')
            print('\r', end='') # use '\r' to go back
            tmp = a.filter(peak_filter, cut[1], cut[2])
            if args.normalizeProteins:
                tmp = [float(x[6])/maxV for x in tmp]
            else:
                tmp = [float(x[6]) for x in tmp]
            if len(tmp) == 0:
                tmp.append(0)
            if args.mergeOperation == 'avg':
                score = np.mean(tmp)
            elif args.mergeOperation == 'max':
                score = np.max(tmp)
            elif args.mergeOperation == 'sum':
                score = np.sum(tmp)
            allProteins[j][i+1] = score 
            j += 1 
        i += 1
    print(args.chrom)
    if args.normalizeProteins:
        nop = "_N"
    else:
        nop = ""
    data = pd.DataFrame(allProteins,columns=['start','ctcf', 'rad21', 'smc3', 'H2az', 'H3k4me1', 'H3k4me2', 'H3k4me3',
                              'H3k9ac', 'H3k9me3', 'H3k27ac', 'H3k27me3',
                               'H3k36me3', 'H3k79me2', 'H4k20me1'],
                               index=range(len(cuts)))
    pickle.dump(data, open(tagCreator(args, "protein") , "wb" ) )

def plotPredMatrix(args):

    name = tagCreator(args, "pred")  
    a = ["--matrix",name,
            "--dpi", "300"]
    if args.log:
        a.extend(["--log1p", "--vMin" ,"1"])
    else:
        a.extend(["--vMax" ,"1","--vMin" ,"0"])
    if args.region:
        a.extend(["--region", args.region])
    elif args.regionIndex1 and args.regionIndex2:
        ma = hm.hiCMatrix(name)
        cuts = ma.cut_intervals
        args.region = args.chrom.split("_")[0] +":"+str(cuts[args.regionIndex1][1])+"-"+ str(cuts[args.regionIndex2][1])
        a.extend(["--region", args.region])

    a.extend( ["-out",tagCreator(args, "image")])
    hicPlot.main(a)

def plotMatrix(args):
    name = args.sourceFile.split(".")[0].split("/")[-1]
    a = ["--matrix",args.sourceFile,
            "--dpi", "300"]
    if args.log:
        a.extend(["--log1p", "--vMin" ,"1","--vMax" ,"1000"])
    else:
        a.extend(["--vMax" ,"1","--vMin" ,"0"])
    if args.region:
        a.extend(["--region", args.region])
        name = name + "_r"+args.region
    elif args.regionIndex1 and args.regionIndex2:
        ma = hm.hiCMatrix(args.sourceFile)
        cuts = ma.cut_intervals
        region = args.chrom +":"+str(cuts[args.regionIndex1][1])+"-"+ str(cuts[args.regionIndex2][1])
        a.extend(["--region", region])
        name = name + "_R"+region

    a.extend( ["-out", IMAGE_D+name+".png"])
    hicPlot.main(a)





def plotAllOfDir(args):
    for f in os.listdir(args.targetDir):
        args.sourceFile = args.targetDir+"/"+f
        plotMatrix(args)




if __name__ == "__main__":
    # main(sys.argv[1:])
    for f in os.listdir(PROTEIN_D):
        if  len(f.split("Proteins_")) > 1:
            n = f.split("Proteins_")[0] 
            m = f.split("Proteins_")[1]
            os.rename(PROTEIN_D+f, PROTEIN_D+n+"M"+m)


