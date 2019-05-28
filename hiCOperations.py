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
BIN_D = "Bin20/"
DATA_D = "Data2e/"
CHROM_D = DATA_D +BIN_D+ "Chroms/"
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


def predictionToMatrix(args):
    pred = pickle.load(open(args.targetDir+ PREDTYPE, "rb" ) )
    ma = hm.hiCMatrix(args.sourceFile)
    mat = ma.matrix.todense()
    new = np.zeros(mat.shape)
    print(new.shape)
    cuts = ma.cut_intervals
    l = len(cuts)
    for j, cut in enumerate(cuts):
        maxV = min(l - j - 1,int(args.reach)+1)
        print(cut[1], maxV)
        for i in range(1,maxV):
            new[j][j+i] = pred.loc[(cut[1], cuts[j+i][1])]['target']
    new = sparse.csr_matrix(new)
    ma.setMatrix(new, ma.cut_intervals)
    newName = "matrix.cool"
    ma.save(args.targetDir + newName)
    # plotMatrix(PRED_D + args.predDir, newName, name=args.predDir)
    # plotMatrix(CHROM_D, args.orig +COOL )




def createForestDataset(args):
    allProteins = pickle.load(open( PROTEIN_D
        +args.chrom+"_allProteinsBinned.p", "rb" )).values.tolist()
        # +args.chrom+"_allProteinsBinned.p", "rb" ))
    # print(allProteins)
    # proLen = allProteins.shape
    # rows = allProteins.shape[0]
    colNr = np.shape(allProteins)[1]
    rows = np.shape(allProteins)[0]
    matrix =args.chrom+"/matrix.cool"
    ma = hm.hiCMatrix(CHROM_D+matrix).matrix
    ma = ma.todense()
    if args.conversion == 'log':
        maxValue = np.amax(ma)
        print(maxValue)
        ma = np.log(ma+1)
        ma /= np.log(maxValue)
    elif args.conversion == 'norm':
        maxValue = np.amax(ma)
        print(maxValue)
        ma /= maxValue
    print(ma.shape)
    print(rows,colNr)
    cols = ['first', 'second','chrom'] + list(range(3* (colNr-1)))+['distance','target']
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
                middleProteins = np.mean(middleProteins, axis=0)[1:]
            frame = [firstStart, secondStart,args.chrom]
            frame.extend(ownProteins)
            frame.extend(middleProteins)
            frame.extend(secondProteins)
            frame.append(distance)
            val = ma[j,j+i]
            frame.append(val)
            window.append(frame)
    data = pd.DataFrame(window,columns =cols)
    print(data.shape)
    # pickle.dump(window, open( SET_D +args.chrom+"_allWindows.p", "wb" ) )
    pickle.dump(data, open( SET_D +args.conversion+"/all/"+args.chrom+".p", "wb" ) )

def createAllWindows(args):
    for i in range(1,21):
        print(i)
        args.chrom = str(i)
        createForestDataset(args)

def loadAllProteins(args):
    for f in os.listdir(CHROM_D):
        loadProtein(args, f)


def loadProtein(args, chromFile=None):
    if chromFile:
        matrix = chromFile
    else:
        matrix =args.chrom+"_Bin"+BIN+".cool"
    print(matrix)
    ma = hm.hiCMatrix(CHROM_D+matrix)
    cuts = ma.cut_intervals
    allProteins = pd.DataFrame(columns=['start','ctcf', 'rad21', 'smc3', 'H2az', 'H3k4me1', 'H3k4me2', 'H3k4me3',
                              'H3k9ac', 'H3k9me3', 'H3k27ac', 'H3k27me3',
                               'H3k36me3', 'H3k79me2', 'H4k20me1'],
                               index=range(len(cuts)))
    i = 0
    for cut in cuts:
        allProteins.iloc[i][0] = cut[1]
        i += 1
    i = 0
    for f in os.listdir(PROTEINORIG_D):
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
            allProteins.iloc[j][i+1] = score
            j += 1 
        i += 1
        print(allProteins.shape)
    c = matrix.split("_")[0]
    print(c)
    pickle.dump(allProteins, open( PROTEIN_D +
                                  c+"_allProteinsBinned.p", "wb" ) )

def plotMatrix(directory, fileName,name=None, log=True):
    if name == None:
        name = fileName
    if log:
        args = ["--matrix",directory + fileName, "-out", IMAGE_D+name+".png",
              "--log1p",   "--dpi", "300", "--vMin" ,"1"]
    else:
        args = ["--matrix",directory + fileName, "-out", IMAGE_D+name+".png",
            "--dpi", "300","--vMax" ,"1","--vMin" ,"0"]
    hicPlot.main(args)


def cutMatrix(args):
    if not os.path.exists(args.targetDir+"Chunks/"):
        os.mkdir(args.targetDir+"Chunks/")
    ma = hm.hiCMatrix(args.targetDir + "matrix.cool")
    matrix = ma.matrix
    matrix = matrix.todense()
    matrix = np.triu(matrix)
    matrix = np.tril(matrix, args.cutWidth-1)
    done = False
    length, width = matrix.shape
    start = 0
    end = args.cutLength
    cuts = ma.cut_intervals
    while(not done):
        print(end)
        if(end >= length):
            end = length - 1
            start = end - args.cutLength
            done = True

        newCuts = cuts[start:end]
        first = newCuts[0][1]
        last = newCuts[-1][2]
        corrCuts = [] 
        for cut in newCuts:
            c = cut[0]
            s = cut[1]
            e = cut[2]
            v = cut[3]
            nCut = (c,s-first, e -first,v)
            corrCuts.append(nCut)
        chunk = matrix[start:end, start:end]
        chunk =  sparse.csr_matrix(chunk)
        region = args.chrom+":"+str(first)+"-"+str(last)
        m = hm.hiCMatrix(None)
        m.setMatrix(chunk,corrCuts)
        region_end =  int(int(region.split("-")[1]) / 10000)
        name = str(region_end).zfill(5) 
        # name = region.split(":")[0] +"_" + str(region_end).zfill(5) 
        m.save(args.targetDir+"Chunks/"+name+".cool")
        start += args.cutLength - args.overlap
        end += args.cutLength - args.overlap

def iterateAll(args):
    allChunks = []
    for f in os.listdir(CHROM_D):
        args.chrom = ma.getChrNames()[0]
        args.targetDir = CHROM_D+args.chrom+"/"
        cutMatrix(args)

def convertMatrix(hicMa, method):
    if method == "customLog":
        customLogVec = np.vectorize(customLog)
        matrix = hicMa.matrix.todense()
        print(matrix[0])
        matrix = customLogVec(matrix)
        print(matrix[0])
        matrix = np.round(matrix, 2)
        matrix = sparse.csr_matrix(matrix)
        hicMa.setMatrix(matrix, hicMa.cut_intervals)
        return hicMa
    if method =='treshold':
        matrix = hicMa.matrix.todense()
        matrix[matrix < 100] = 10
        # matrix = np.log(matrix) / np.log(20000)
        matrix = sparse.csr_matrix(matrix)
        hicMa.setMatrix(matrix, hicMa.cut_intervals)
        return hicMa
    if method =='manualAssign':
        manualVec = np.vectorize(manualAssign)
        matrix = hicMa.matrix.todense()
        print(matrix[0])
        matrix = manualVec(matrix)
        matrix = sparse.csr_matrix(matrix)
        hicMa.setMatrix(matrix, hicMa.cut_intervals)
        return hicMa

def convertAll(chromosome):
    i = 0
    d = CHUNK_D
    for f in os.listdir(d):
        c = f.split("_")[0]
        e = f.split("_")[1]
        if chromosome == int(c):
        # if "00205.cool" == e
                ma = hm.hiCMatrix(d+f)
                # ma = convertMatrix(ma, 'customLog')
                ma.save(CUSTOM_D  + f)
                i += 1
                if i > 10:
                    break

def plotAllOfDir(args):
    for f in os.listdir(args.targetDir):
        x = f.split(".")[0]
        plotMatrix(args.targetDir,f,args.prefix+x+args.suffix, args.log )

def printAll(chromosome, d):
    i = 0
    for f in os.listdir(d):
        c = f.split("_")[0]
        e = f.split("_")[1]
        # if "00205.cool" == e:
        if chromosome == int(c):
                plotMatrix(d,f, False)
                # ma = hm.hiCMatrix(d+f)
                i += 1
                if i > 10:
                    break

def reverseCustomLog(a):
    if a == 0:
        return 0
    elif a > reverseMiddle:
        a += sink
        a *= np.log(maxLog)
        a = np.exp(a)
        a += shift
        return a
    else:
        return (a**(1/exponent))*divisor

sink = 0
logMax = 10000
reverseMiddle = 0
shift = 0 #32.5
middle = 0
divisor = 1#50
exponent =1#5

def manualAssign(a):
    if a < 40: return 0
    elif a < 80: return 0.1
    elif a < 150: return 0.2
    elif a < 250: return 0.3
    elif a < 400: return 0.4
    elif a < 600: return 0.5
    elif a < 800: return 0.6
    elif a < 1500: return 0.7 
    elif a < 2500: return 0.8
    else: return 0.9

def customLog(a):
    if a == 0:
        return 0
    elif a >= middle:
        a -= shift
        return np.log(a) /np.log(logMax) - sink
    else:
        return (a/divisor)**exponent


def createDataset(args, create_test =False):
    if create_test:
        d = TEST_D
    else:
        d = CHUNK_D
    matrixList = []
    customLogVec = np.vectorize(customLog)
    for f in os.listdir(d):
        c = f.split("_")[0]
        if args.chrom == c:
            print(f)
            ma = hm.hiCMatrix(d+f)
            matrix = ma.matrix.todense()
            matrix = np.triu(matrix)
            safe = deepcopy(matrix)

            maxV = matrix.max()
            if np.isnan(matrix).any() or maxV == 0:
                continue
            matrix[matrix < args.treshold] = 0
            if args.prepData == 'customLog':
                print(matrix[0])
                matrix = customLogVec(matrix)
                print(matrix[0])
            elif args.prepData == 'log':
                matrix += 1
                matrix = np.log(matrix) /np.log(args.maxValue)
            else: 
                matrix = np.asarray(matrix)
            matrixList.append(matrix)
    else:
        tensor_x = torch.cat(tuple(torch.Tensor([[i]]) for i in matrixList))
    print(tensor_x.shape)
    dataset = utils.TensorDataset(tensor_x)
    test =""
    if create_test:
        test = "_test"
    tre = "_t"+str(args.treshold)
    pickle.dump(dataset, open( SET_D + args.chrom+tre+"_"+args.prepData+test+".p", "wb" ) )

def parseArguments(args=None):
    print(args)

    parser = argparse.ArgumentParser(description='HiC Prediction')

    parserRequired = parser.add_argument_group('Required arguments')

    # define the arguments
    parserRequired.add_argument('--action', '-a', choices=[ 'create','cutMatrix', 
                    'createAverage','loadAllProteins','loadProtein','predToM','cutAll',
                    'createAllWindows','plotAll','createWindows'], help='Action to take', required=True)

    parserOpt = parser.add_argument_group('Optional arguments')
    parserOpt.add_argument('--treshold', '-tr',type=int, default=0)
    parserOpt.add_argument('--mergeOperation', '-mo',type=str, default='avg')
    parserOpt.add_argument('--chrom', '-c',type=str, default="1")
    parserOpt.add_argument('--bin', '-b',type=str, default="20")
    parserOpt.add_argument('--reach', '-r',type=str, default="99")
    parserOpt.add_argument('--conversion', '-co',type=str, default="default")
    parserOpt.add_argument('--suffix', '-s',type=str, default="")
    parserOpt.add_argument('--prefix', '-p',type=str, default="")
    parserOpt.add_argument('--sourceFile', '-sf',type=str, default="")
    parserOpt.add_argument('--targetDir', '-td',type=str, default="")
    parserOpt.add_argument('--cutLength', '-cl',type=int, default=100)
    parserOpt.add_argument('--overlap', '-o',type=int, default=0)
    parserOpt.add_argument('--cutWidth', '-cw',type=int, default=100)
    parserOpt.add_argument('--maxValue', '-mv',type=int, default=10068)
    parserOpt.add_argument('--validationData', '-v',type=bool, default=False)
    parserOpt.add_argument('--log', '-l',type=bool, default=False)
    parserOpt.add_argument('--trainData', '-t',type=bool,default=True)
    args = parser.parse_args(args)
    print(args)
    return args

def main(args=None):
    args = parseArguments(args)
    if args.action == "create":
        createDataset(args, True)
        createDataset(args, False)
    elif args.action == "cutAll":
        iterateAll(args)
    elif args.action == "cutMatrix":
        cutMatrix(args)
    elif args.action == "loadProtein":
        loadProtein(args, chromFile=None)
    elif args.action == "loadAllProteins":
        loadAllProteins(args)
    elif args.action == "createWindows":
        createForestDataset(args)
    elif args.action == "createAllWindows":
        createAllWindows(args)
    elif args.action == "predToM":
        predictionToMatrix(args)
    elif args.action == "plotAll":
        plotAllOfDir(args)


if __name__ == "__main__":
    main(sys.argv[1:])

