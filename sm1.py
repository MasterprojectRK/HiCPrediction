from hicmatrix import HiCMatrix as hm
import torch
from hicexplorer import hicPlotMatrix as hicPlot
import numpy as np
from scipy import sparse
import os
import logging as log
import sys
from pprint import pprint
from collections import deque
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import torch
import torch.utils.data as utils
import argparse
import torch.multiprocessing as mp
import pickle
from m1 import *
import cooler
from copy import copy, deepcopy

log.basicConfig(level=log.DEBUG)
np.set_printoptions(threshold=sys.maxsize)
torch.set_num_threads(16)
np.set_printoptions(precision=3, suppress=True)


def plotMatrix(directory, fileName):
    name = os.path.splitext(fileName)[0]
    args = ["--matrix",directory + fileName, "-out", IMAGE_D+name+".png",
            "--log1p", "--dpi", "300"]
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

def printAll(chromosome):
    d = TEST_D
    for f in os.listdir(d):
        c = f.split("_")[0]
        if chromosome == int(c):
                plotMatrix(d,f)
                ma = hm.hiCMatrix(d+f)

def reverseCustomLog(a):
    if a == 0:
        return 0
    elif a > 0.25:
        a *= np.log(10000)
        a = np.exp(a)
        a += 32.5
        return a
    else:
        return (a**0.2)*50



def customLog(a):
    if a == 0:
        return 0
    elif a >= 38:
        a -= 32.5
        return np.log(a) /np.log(10000)
    else:
        return (a/50)**5


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


def applyAE(args, test=False): 
    print(args)
    customLogVec = np.vectorize(customLog)
    reverseCustomLogVec = np.vectorize(reverseCustomLog)
    model =SparseAutoencoder(args)
    model.load_state_dict(torch.load(MODEL_D+args.model))
    model.eval()
    if test:
        d = TEST_D
        end = "11038"
    else:
        d = CHUNK_D 
        end = "15238"
    name = args.chrom + "_"+end
    matrix = name +".cool"
    ma = hm.hiCMatrix(d+matrix)
    m = ma.matrix.todense()
    m = np.triu(m)
    m[m < args.treshold] = 0
    print(m[0])
    # print(m.shape)
    newCus = deepcopy(m)
    if args.prepData == 'log':
        m += 1
        m = np.log(m) /  np.log(args.maxValue)
    elif args.prepData == 'customLog':
        m= customLogVec(m)
        newCus= customLogVec(newCus)
    else:
        m = m.tolist()
    print(m[0])
    # print(m[10])
    t = torch.Tensor([[m]])
    encoded, decoded = model(t)
    # print(encoded)
    decoded = decoded[0][0]
    encoded = encoded[0]
    # en = encoded.detach().numpy().reshape(100,100)
    # print(en)
    new = decoded.detach().numpy()
    new2 = deepcopy(new)
    # print(np.sum(abs(m-new))/10000)
    # print(new[10])
    print(new[0])
    if args.prepData == 'customLog' or args.prepData == 'log':
        new *= np.log(args.maxValue)
        new = np.exp(new)
        new -= 1
        # en *= np.log(args.maxValue)
        # en = np.exp(en)
        # en -= 1
    if args.prepData == 'customLog':
        new2 = reverseCustomLogVec(new2)
    print(new[0])
    new = sparse.csr_matrix(new)
    newCus = sparse.csr_matrix(newCus)
    new2 = sparse.csr_matrix(new2)
    # plotMatrix(d,matrix)
    print(name)
    ma.setMatrix(new, ma.cut_intervals)
    ma.save(PRED_D + name + "_P_L1.cool")
    plotMatrix(PRED_D, name + "_P_L1.cool")
    if args.prepData == 'customLog':
        ma.setMatrix(new2, ma.cut_intervals)
        ma.save(PRED_D + name + "_P_L1_Own.cool")
        plotMatrix(PRED_D, name + "_P_L1_Own.cool")
        # ma.setMatrix(newCus, ma.cut_intervals)
    # ma.save(PRED_D + name + "_Cus.cool")
    # plotMatrix(PRED_D, name + "_Cus.cool")

def startTraining(args):
    model = SparseAutoencoder(args)
    return train(model,args)

def parseArguments(args=None):
    print(args)

    parser = argparse.ArgumentParser(description='HiC Prediction')

    parserRequired = parser.add_argument_group('Required arguments')

    # define the arguments
    parserRequired.add_argument('--action', '-a', choices=['train', 'create',
         'predict','cutMatrix', 'createAverage'], help='Action to take', required=True)

    parserOpt = parser.add_argument_group('Optional arguments')
    parserOpt.add_argument('--learningRate', '-lr',type=float, default=0.001)
    parserOpt.add_argument('--epochs', '-e',type=int, default=1000)
    parserOpt.add_argument('--treshold', '-tr',type=int, default=0)
    parserOpt.add_argument('--beta', '-be',type=float, default=0.1)
    parserOpt.add_argument('--chrom', '-c',type=str, default="4")
    parserOpt.add_argument('--cutLength', '-cl',type=int, default=100)
    parserOpt.add_argument('--hidden', '-hl',type=str, default="S")
    parserOpt.add_argument('--output', '-ol',type=str, default="S")
    parserOpt.add_argument('--outputSize', '-os',type=int, default=1000)
    parserOpt.add_argument('--overlap', '-o',type=int, default=50)
    parserOpt.add_argument('--cutWidth', '-cw',type=int, default=100)
    parserOpt.add_argument('--maxValue', '-mv',type=int, default=10068)
    parserOpt.add_argument('--convSize', '-cs',type=int, default=5)
    parserOpt.add_argument('--firstLayer', '-fl',type=int, default=4)
    parserOpt.add_argument('--secondLayer', '-sl',type=int, default=8)
    parserOpt.add_argument('--thirdLayer', '-tl',type=int, default=16)
    parserOpt.add_argument('--dropout1', '-d1',type=float, default=0.1)
    parserOpt.add_argument('--pool1', '-p1',type=int, default=2)
    parserOpt.add_argument('--dimAfterP1', '-dim1',type=int, default=0)
    parserOpt.add_argument('--dimAfterP2', '-dim2',type=int, default=0)
    parserOpt.add_argument('--dropout2', '-d2',type=float, default=0.3)
    parserOpt.add_argument('--padding', '-p',type=int, default=0)
    parserOpt.add_argument('--batchSize', '-b',type=int, default=256)
    parserOpt.add_argument('--model', '-m',type=str, default='autoencoder.pt')
    parserOpt.add_argument('--saveModel', '-sm', default=True)
    parserOpt.add_argument('--loss', '-l',type=str, default='L1')
    parserOpt.add_argument('--prepData', '-dp',type=str, default='log')
    parserOpt.add_argument('--validationData', '-v',type=bool, default=False)
    parserOpt.add_argument('--lastLayer', '-ll',type=str, default='third')
     
    parserOpt.add_argument('--trainData', '-t',type=bool,default=True)
    args = parser.parse_args(args)
    args.dimAfterP1 = int(args.cutWidth / args.pool1)
    args.dimAfterP2 = int(args.dimAfterP1 / args.pool1)
    args.padding = int(np.floor(args.convSize / 2)) 
    print(args)
    return args

def main(args=None):
    args = parseArguments(args)
    if args.action == "train":
        startTraining(args)
    elif args.action == "create":
        createDataset(args, True)
        createDataset(args, False)
    elif args.action == "predict":
        applyAE(args, test=True)
        applyAE(args,  test=False)
    elif args.action == "cutMatrix":
        iterateAll(args)
    elif args.action == "createAverage":
        CHROM = args.chrom
        MAX =args.maxValue
        createAverage()


if __name__ == "__main__":
    main(sys.argv[1:])
    # printAll(4)
    # d = CHUNK_C
    # end = "15238"
    # name = CHR + "_"+end
    # matrix = name +".cool"
    # ma = hm.hiCMatrix(d+matrix)
    # m = ma.matrix.todense()
    # m[m<15] = 0
    # print(m[0])
    # new = sparse.csr_matrix(m)
    # ma.setMatrix(new, ma.cut_intervals)
    # ma.save(PRED_D + name + "_treshold15.cool")
    # plotMatrix(PRED_D, name + "_treshold15.cool")
