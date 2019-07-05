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
BIN_D = "5B/"
binSize = 5000
DATA_D = "Data2e/"
RESULT_D = "Data2e/Results/"
CHROM_D = DATA_D +BIN_D+ "Chroms/"
ARM_D = DATA_D +BIN_D+ "Arms/"
SET_D = DATA_D + BIN_D +"Sets/"
SETC_D = DATA_D + BIN_D +"SetsCombined/"
PRED_D = DATA_D +BIN_D+ "Predictions/"
MODEL_D  = DATA_D + BIN_D+"Models/"
IMAGE_D = DATA_D +  "Images/"
PLOT_D = DATA_D +BIN_D +  "Plots/"
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


def predictionPreparation(args, pred=1):
    # if pred == 1:
        # pred = pickle.load(open(tagCreator(args, "pred"), "rb" ) )
    if len(args.chrom.split("_")) > 1:
        predictionToMatrix(args, pred)
    else:
        print(tagCreator(args, "pred"))
        c = args.chrom
        args.chrom = c +"_A"
        predA = pred.loc[pred['chrom'] == args.chrom]
        if  os.path.isfile(ARM_D +args.chrom+".cool")\
            and not os.path.isfile(tagCreator(args,"pred")):
            predictionToMatrix(args, predA)
        else: 
            print("A exists")
        
        args.chrom = c +"_B"
        predB = pred.loc[pred['chrom'] == args.chrom]
        if  os.path.isfile(ARM_D +args.chrom+".cool")\
            and not os.path.isfile(tagCreator(args,"pred")):
            predictionToMatrix(args, predB)
        else: 
            print("B exists")



def predictionToMatrix(args, pred):
    print("Now Converting:")
    ma = hm.hiCMatrix(ARM_D +args.chrom+".cool")
    mat = ma.matrix.todense()
    factor =  np.max(mat)
    new = np.zeros(mat.shape)
    cuts = ma.cut_intervals
    l = len(cuts)
    for j, cut in enumerate(cuts):
        maxV = min(l - j - 1,int(args.reach)+1)
        print(str(j+1)+"/"+str(len(cuts)),end='')
        print('\r', end='') # use '\r' to go back
        sys.stdout.flush()

        for i in range(0,maxV):
            # print(i)
            # print(j)
            # print(cut)
            # print(cuts[j+i])
            # print(pred[:2])
            val = pred.loc[(cut[1], cuts[j+i][1])]['pred']
            if args.conversion == "norm":
                val= val  * factor
            elif args.conversion == "log":
                val= val  * np.log(factor)
                val = np.exp(val) - 1
            elif args.conversion == "standardLog":
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
            continue
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
    for i in range(1,13):
        args.chrom = str(i)
        divideIntoArms(args)
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
    csam = csa + "_"+args.model +"_" + args.loss

    if mode == "set":
        return SET_D + args.chrom + wmep +".p"

    elif mode == "model":
        return MODEL_D + csam + cowmep+".p"

    elif mode == "protein":
        return PROTEIN_D + args.chrom+ "_M"+args.mergeOperation+nop+".p"
    elif mode == "pred":
        return PRED_D + args.chrom + "_P"+ csam + cowmep +".cool"

    elif mode == "setC":
        return SETC_D+csa+ wmep +".p"

    elif mode == "image":
        return IMAGE_D + args.chrom +"_R" + args.region+"_P"+csam + cowmep +".png"
    elif mode == "plot":
        return PLOT_D  + args.chrom +"_P"+csam + cowmep +".png"

def createForestDataset(args):
    allProteins = pickle.load(open(tagCreator(args, "protein"), "rb" )).values.tolist()
    colNr = np.shape(allProteins)[1]
    zeroProteins = np.zeros(colNr -1)
    rows = np.shape(allProteins)[0]
    ma = hm.hiCMatrix(ARM_D+args.chrom+".cool").matrix
    ma2 = hm.hiCMatrix(CHROM_D+args.chrom.split("_")[0]+".cool").matrix
    reads = ma.todense()
    logs = deepcopy(reads)
    norms = deepcopy(reads)
    maxValue = np.max(ma2.todense()) 
    logs = np.log(logs+1)
    logs /= np.log(maxValue)
    norms =norms /  maxValue
    print(ma.shape)
    print(rows,colNr)
    cols = ['first', 'second','chrom'] + list(range(3*
         (colNr-1)))+['distance','reads','logTarget','normTarget']
    window = []
    if args.windowOperation == 'avg':
        convertF = np.mean
    elif args.windowOperation == 'sum':
        convertF = np.sum
    elif args.windowOperation == 'max':
        convertF = np.max
    for j in range(rows):
        maxReach = min(int(args.reach)+1,rows-j)
        firstStart = allProteins[j][0]
        for  i in range(0,maxReach):
            if j+1 >= j+i:
                middleProteins = zeroProteins
            else:
                middleProteins = convertF(allProteins[j+1:j+i], axis=0)[1:]
            frame = [firstStart, allProteins[j+i][0],args.chrom]
            frame.extend(allProteins[j][1:])
            frame.extend(middleProteins)
            frame.extend(allProteins[j+i][1:])
            frame.extend([i*binSize, reads[j,j+i],logs[j,j+i],norms[j,j+i]])
            window.append(frame)
    data = pd.DataFrame(window,columns =cols)
    print(data.shape)
    pickle.dump(data, open(tagCreator(args, "set"), "wb" ) )


def createAllCombinations(args):
    for w in ["avg"]:
        args.windowOperation = w
        for m in ["avg"]:
            args.mergeOperation = m
            for n in [False]:
                args.normalizeProteins = n
                for e in [False]:
                    args.equalizeProteins = e
                    createAllWindows(args)


def createAllWindows(args):
    i = 1
    for f in os.listdir(ARM_D):
        args.chrom = f.split(".")[0]
        if  os.path.isfile(tagCreator(args, "protein")):
            # if int(args.chrom.split("_")[0]) < 22:
                # continue
            print(args.chrom)
            if  not os.path.isfile(tagCreator(args, "set")):
                createForestDataset(args)

def loadAllProteins(args):
    for f in os.listdir(ARM_D):
        args.chrom = f.split(".")[0]
        c = int(f.split("_")[0])
        if  not os.path.isfile(tagCreator(args, "protein")):
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
        minV = min(c)
        maxV = max(c) - minV
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
                tmp = [(float(x[6]) -minV) / maxV for x in tmp]
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
    print(a)
    args.region = None
    hicPlot.main(a)

def plotMatrix(args):
    # for i in range(10):
        # args.regionIndex1 = i*200 + 1
        # args.regionIndex1 = (i+1)*200
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


def plotDir(args):
    for cs in [9,11,14,17,19]:
        args.chroms = str(cs)
        for c in ["9_A","9_B","11_A", "11_B","14_A", "14_B","17_A",
                  "17_B","19_A", "19_B"]:
            args.chrom = str(c)
            for p in ["default", "standardLog"]:
                args.conversion = p
                for w in ["avg"]:
                    args.windowOperation = w
                    for me in ["avg"]:
                        args.mergeOperation = me
                        for m in ["rf"]:
                            args.model = m
                            for n in [False, True]:
                                args.normalizeProteins = n
                                for n in [False, True]:
                                    args.equalizeProteins = n
                                    if os.path.isfile(tagCreator(args,"pred")):
                                        print(tagCreator(args, "pred"))
                                        for i in range(10):
                                            args.regionIndex1 = i*200 + 1
                                            args.regionIndex2 = (i+1)*200
                                            plotPredMatrix(args)






if __name__ == "__main__":
    d = PRED_D
    # key = "rf_n"
    # for f in os.listdir(d):
        # if len(f.split(key))>1:
            # n = f.split(key)[0]
            # m = f.split(key)[1]
            # os.rename(d+f, d+n+"rf_mse_n"+m)
