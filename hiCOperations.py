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
CHR = "4"
BIN = "20"
DATA_D = "Data2e/"
CHROM_D = DATA_D +BIN+ "Chroms/"
SET_D = DATA_D + BIN +"Sets/"
PRED_D = DATA_D + "Predictions/"
MODEL_D  = DATA_D + "Models/"
CHUNK_D = DATA_D +BIN +"Chunks/"
CUSTOM_D = DATA_D +BIN +"Custom/"
TEST_D =  DATA_D +BIN+ "Test/"
IMAGE_D = DATA_D +  "Images/"
ORIG_D = DATA_D +  "Orig/"
PROTEIN_D = DATA_D + BIN+"Proteins/"
allProteins = pd.DataFrame()
def chrom_filter(feature, c):
        "Returns True if correct chrom"
        return feature.chrom == c

def peak_filter(feature, s, e):
        "Returns True if correct chrom"
        peak = feature.start + int(feature[9])
        return s < peak and e > peak

def createForestDataset(args):
    allProteins = pickle.load(open( PROTEIN_D +args.chrom+"_allProteinsBinned.p", "rb" ) )
    proLen = allProteins.shape[1]
    rows = allProteins.shape[0]
    matrix ="4_Bin20.cool"
    ma = hm.hiCMatrix(CHROM_D+matrix).matrix
    ma = ma.todense()
    print(ma.shape)
    print(rows,proLen)
    cols = ['first', 'second'] + list(range(3*(proLen-1)))+['distance','target']
    # window.set_index('firstSecond', inplace=True)
    # step = int(rows/30)
    # for l in range(31):
        # s = l * step 
        # e = min((l+1)*step, rows)
        # window = pd.DataFrame(columns =cols)
        # k = 0
        # for j in range(s,e):
    window = pd.DataFrame(columns =cols)
    k = 0
    for j in range(rows):
        maxReach = min(int(args.reach)+1,rows-j)
        ownProteins = allProteins[allProteins.columns.difference(['start'])].iloc[j]
        ownProteins = ownProteins.rename('first').values.tolist()
        firstStart = allProteins.iloc[j]['start']
        for  i in range(1,maxReach):
            secondStart = allProteins.iloc[j+i]['start']
            distance = secondStart - firstStart
            secondProteins = allProteins[allProteins.columns.difference(['start'])].iloc[j+i]
            secondProteins = secondProteins.rename('second').values.tolist()
            middleProteins = allProteins[allProteins.columns.difference(['start'])].iloc[j+1:j+i]
            middleProteins = middleProteins.mean(axis=0).rename('middle').values.tolist()
            frame = [firstStart, secondStart]
            frame.extend(ownProteins)
            frame.extend(middleProteins)
            frame.extend(secondProteins)
            frame.append(distance)
            val = ma[j,j+i]
            frame.append(val)
            index = str(j) +"_"+str(i)
            window.loc[k] = frame
            k += 1
        print(window.shape)
    pickle.dump(window, open( SET_D +args.chrom+"_allWindows.p", "wb" ) )
    # pickle.dump(window, open( SET_D +args.chrom+"_windows"+str(l)+".p", "wb" ) )

    def loadProteins(args):
        matrix ="4_Bin20.cool"
        ma = hm.hiCMatrix(CHROM_D+matrix)
        cuts = ma.cut_intervals
        allProteins = pd.DataFrame(columns=['start', 'H2az', 'H3k4me1', 'H3k4me2', 'H3k4me3',
                                  'H3k9ac', 'H3k9me3', 'H3k27ac', 'H3k27me3',
                                   'H3k36me3', 'H3k79me2', 'H4k20me1'],
                                   index=range(len(cuts)))
        i = 0
    for cut in cuts:
        allProteins.iloc[i][0] = cut[1]
        i += 1
    i = 0
    for f in os.listdir(PROTEIN_D):
        path  = PROTEIN_D+f
        a = pybedtools.BedTool(path)
        a = a.filter(chrom_filter, c='chr'+CHR)
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
    print(allProteins)
    pickle.dump(allProteins, open( PROTEIN_D +
                                  args.chrom+"_allProteinsBinned.p", "wb" ) )

def plotMatrix(directory, fileName, log):
    name = os.path.splitext(fileName)[0]
    if log:
        args = ["--matrix",directory + fileName, "-out", IMAGE_D+name+".png",
              "--log1p",   "--dpi", "300", "--vMin" ,"1"]
    else:
        args = ["--matrix",directory + fileName, "-out", IMAGE_D+name+".png",
            "--dpi", "300","--vMax" ,"1","--vMin" ,"0"]
    hicPlot.main(args)


def cutMatrix(ma,chrom, args):
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
        region = chrom+":"+str(first)+"-"+str(last)
        m = hm.hiCMatrix(None)
        m.setMatrix(chunk,corrCuts)
        region_end =  int(int(region.split("-")[1]) / 10000)
        name = region.split(":")[0] +"_" + str(region_end).zfill(5) 
        m.save(CHUNK_D+name+".cool")
        start += args.cutLength - args.overlap
        end += args.cutLength - args.overlap

def iterateAll(args):
    allChunks = []
    for f in os.listdir(CHROM_D):
        ma = hm.hiCMatrix(CHROM_D+f)
        print(f, ma.matrix.shape)
        cutMatrix(ma, ma.getChrNames()[0],args)

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
                    'createAverage', 'loadProteins', 'createWindows'], help='Action to take', required=True)

    parserOpt = parser.add_argument_group('Optional arguments')
    parserOpt.add_argument('--treshold', '-tr',type=int, default=0)
    parserOpt.add_argument('--mergeOperation', '-mo',type=str, default='sum')
    parserOpt.add_argument('--chrom', '-c',type=str, default="4")
    parserOpt.add_argument('--reach', '-r',type=str, default="100")
    parserOpt.add_argument('--cutLength', '-cl',type=int, default=100)
    parserOpt.add_argument('--overlap', '-o',type=int, default=40)
    parserOpt.add_argument('--cutWidth', '-cw',type=int, default=100)
    parserOpt.add_argument('--maxValue', '-mv',type=int, default=10068)
    parserOpt.add_argument('--validationData', '-v',type=bool, default=False)
    parserOpt.add_argument('--trainData', '-t',type=bool,default=True)
    args = parser.parse_args(args)
    print(args)
    return args

def main(args=None):
    args = parseArguments(args)
    if args.action == "create":
        createDataset(args, True)
        createDataset(args, False)
    elif args.action == "cutMatrix":
        iterateAll(args)
    elif args.action == "loadProteins":
        loadProteins(args)
    elif args.action == "createWindows":
        createForestDataset(args)


if __name__ == "__main__":
    main(sys.argv[1:])
    # convertAll(4)
    # printAll(4, CUSTOM_D)
    # d = CHUNK_D
    # end = "00205"
    # name = CHR + "_"+end
    # matrix = name +".cool"
    # ma = hm.hiCMatrix(d+matrix)
    # ma.save(PRED_D + name + "_orig.cool")
    # m = ma.matrix.todense()
    # # # m[m<15] = 0
    # # print(m[0])
    # m = np.log(m) /np.log(20000)

    # new = sparse.csr_matrix(m)
    # ma.setMatrix(new, ma.cut_intervals)
    # ma.save(PRED_D + name + "_log.cool")
    # # m = np.square(m)
    # # m = np.square(m)
    # p = 1
    # q = 4
    # j = 16
    # scharr = np.array([[ q, p,q],
            # [p, j, p],
            # [q, p,  q]]) # Gx + j*Gy
    # m = signal.convolve2d(m, scharr, mode='same')
    # new = sparse.csr_matrix(m)
    # ma.setMatrix(new, ma.cut_intervals)
    # ma.save(PRED_D + name + "_log_log.cool")
    # plotMatrix(PRED_D, name + "_log.cool", False)
    # plotMatrix(PRED_D, name + "_log_log.cool", False)
    # plotMatrix(PRED_D, name + "_orig.cool", False)
