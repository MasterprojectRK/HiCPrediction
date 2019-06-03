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
        return s < peak and e > peak


def predictionToMatrix(args, pred=None):
    if pred is None:
        pred = pickle.load(open(PRED_D+tagCreator(args, "pred")+".p", "rb" ) )
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

        for i in range(maxV):
            val = pred.loc[(cut[1], cuts[j+i][1])]
            if args.conversion == "norm":
                val= val['pred']  * factor
            elif args.conversion == "log":
                val= val['pred']  * np.log(factor)
                val = np.exp(val) - 1
            new[j][j+i] = val
    new = sparse.csr_matrix(new)
    ma.setMatrix(new, ma.cut_intervals)
    newName = tagCreator(args, "pred")+".cool"
    ma.save(PRED_D + newName)
    # plotMatrix(PRED_D + args.predDir, newName, name=args.predDir)
    # plotMatrix(CHROM_D, args.orig +COOL )

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
    end =  "_"+args.conversion + "_"+args.windowOperation
    if mode == "set":
        return args.chrom + "_" +args.windowOperation 
    elif mode == "model":
        return chromsToName(args.chroms)+"_"+args.arms+ end
    elif mode == "pred":
        return args.chrom + "P_"+chromsToName(args.chroms)+"_"+args.arms+ end 
    elif mode == "setC":
        return chromsToName(args.chroms)+"_"+args.arms+ "_"+ args.windowOperation

def createForestDataset(args):
    allProteins = pickle.load(open( PROTEIN_D
        +args.chrom+"_Proteins.p", "rb" )).values.tolist()
        # +args.chrom+"_allProteinsBinned.p", "rb" ))
    # print(allProteins)
    # proLen = allProteins.shape
    # rows = allProteins.shape[0]
    colNr = np.shape(allProteins)[1]
    rows = np.shape(allProteins)[0]
    ma = hm.hiCMatrix(ARM_D+args.chrom+".cool").matrix
    reads = ma.todense()
    logs = deepcopy(reads)
    norms = deepcopy(reads)
    
    maxValue = np.amax(logs)
    print(maxValue)
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
        for  i in range(0,maxReach):
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
    pickle.dump(data, open( SET_D+tagCreator(args, "set")+".p", "wb" ) )



def createAllWindows(args):
    for f in os.listdir(ARM_D):
        args.chrom = f.split(".")[0]
        createForestDataset(args)

def loadAllProteins(args):
    for f in os.listdir(ARM_D):
        args.chrom = f.split("_")[0]
        loadProtein(args, f)


def loadProtein(args, matrix=None):
    print(matrix)
    ma = hm.hiCMatrix(ARM_D+matrix)
    cuts = ma.cut_intervals
    i = 0
    allProteins = []
    for cut in cuts:
        allProteins.append(np.zeros(15))
        allProteins[i][0] = cut[1]
        i += 1
    i = 0
    for f in os.listdir(PROTEINORIG_D):
        print(i)
        path  = PROTEINORIG_D+f
        a = pybedtools.BedTool(path)
        a = a.filter(chrom_filter, c='chr'+args.chrom)
        a = a.sort()
        j = 0
        for cut in cuts:    
            tmp = a.filter(peak_filter, cut[1], cut[2])
            tmp = [float(x.score) for x in tmp]
            if args.mergeOperation == 'sum':
                score = sum(tmp)
            allProteins[j][i+1] = score
            j += 1 
        i += 1
    c = matrix.split(".")[0]
    print(c)
    data = pd.DataFrame(allProteins,columns=['start','ctcf', 'rad21', 'smc3', 'H2az', 'H3k4me1', 'H3k4me2', 'H3k4me3',
                              'H3k9ac', 'H3k9me3', 'H3k27ac', 'H3k27me3',
                               'H3k36me3', 'H3k79me2', 'H4k20me1'],
                               index=range(len(cuts)))
    pickle.dump(data, open( PROTEIN_D + c+"_Proteins.p", "wb" ) )

def plotMatrix(args):

    name = args.sourceFile.split(".")[0].split("/")[-1]
    a = ["--matrix",args.sourceFile,
            "--dpi", "300"]
    if args.log:
        a.extend(["--log1p", "--vMin" ,"1"])
    else:
        a.extend(["--vMax" ,"1","--vMin" ,"0"])
    if args.region:
        a.extend(["--region", args.region])
        name = name + "_r"+args.region
    a.extend( ["-out", IMAGE_D+name+".png"])
    hicPlot.main(a)





def plotAllOfDir(args):
    for f in os.listdir(args.targetDir):
        x = f.split(".")[0]
        print(args.targetDir,f)
        plotMatrix(args.targetDir,f,x, args.log )
                #FORGOT NORMAL PLOT FOR NORMAL MATRIX

# def plotRegion(args):
    # model = chromsToName(args.chroms)
    # targetDir = PRED_D+args.conversion+ "/" +args.chrom +"P/" +model
    # plotMatrix(targetDir,'matrix.cool',args.chrom+"_"+args.region +"_"+
        # args.chroms+args.conversion, args.log )


def printCuts(args):
    ma = hm.hiCMatrix(args.sourceFile)
    print(ma.cut_intervals)


def parseArguments(args=None):
    print(args)

    parser = argparse.ArgumentParser(description='HiC Prediction')

    parserRequired = parser.add_argument_group('Required arguments')

    # define the arguments
    parserRequired.add_argument('--action', '-a', choices=[ 'createAllWindows','plotAll',
    'plot','createArms', 'printCuts','loadAllProteins' ,  'createWindows'],
        help='Action to take', required=True)

    parserOpt = parser.add_argument_group('Optional arguments')
    # parserOpt.add_argument('--treshold', '-tr',type=int, default=0)
    parserOpt.add_argument('--mergeOperation', '-mo',type=str, default='sum')
    parserOpt.add_argument('--windowOperation', '-wo',type=str, default='avg')
    parserOpt.add_argument('--chrom', '-c',type=str, default="1")
    # parserOpt.add_argument('--bin', '-b',type=str, default="20")
    parserOpt.add_argument('--reach', '-r',type=str, default="99")
    parserOpt.add_argument('--conversion', '-co',type=str, default="default")
    parserOpt.add_argument('--chroms', '-cs',type=str, default="1_2")
    # parserOpt.add_argument('--suffix', '-s',type=str, default="")
    parserOpt.add_argument('--region', '-re',type=str, default=None)
    parserOpt.add_argument('--arms', '-ar',type=str, default="AB")
    parserOpt.add_argument('--name', '-n',type=str, default="")
    parserOpt.add_argument('--sourceFile', '-sf',type=str, default="")
    parserOpt.add_argument('--targetDir', '-td',type=str, default="")
    # parserOpt.add_argument('--cutLength', '-cl',type=int, default=100)
    # parserOpt.add_argument('--overlap', '-o',type=int, default=0)
    # parserOpt.add_argument('--cutWidth', '-cw',type=int, default=100)
    # parserOpt.add_argument('--maxValue', '-mv',type=int, default=10068)
    # parserOpt.add_argument('--validationData', '-v',type=bool, default=False)
    parserOpt.add_argument('--log', '-l',type=bool, default=True)
    # parserOpt.add_argument('--trainData', '-t',type=bool,default=True)
    args = parser.parse_args(args)
    print(args)
    return args

def main(args=None):
    args = parseArguments(args)
    if args.action == "loadProtein":
        loadProtein(args, chromFile=None)
    elif args.action == "loadAllProteins":
        loadAllProteins(args)
    elif args.action == "createWindows":
        createForestDataset(args)
    elif args.action == "createAllWindows":
        createAllWindows(args)
    # elif args.action == "predToM":
        # predictionToMatrix(args)
    elif args.action == "plot":
        plotMatrix(args)
    elif args.action == "plotAll":
        plotAllOfDir(args)
    elif args.action == "createArms":
        createAllArms(args)
    elif args.action == "printCuts":
        printCuts(args)


if __name__ == "__main__":
    main(sys.argv[1:])

