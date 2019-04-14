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
import pickle
from sparse_AE import *
import cooler
from copy import copy, deepcopy

log.basicConfig(level=log.DEBUG)
np.set_printoptions(threshold=sys.maxsize)


def createAverage():
    x = np.zeros([200,200]) 
    t =  pickle.load( open(SET_D+CHR+"_test.p","rb" ) )
    for y in t:
        x = x + Variable(y[0])[0].detach().numpy()
    v = pickle.load( open(SET_D+CHR+".p","rb" ) )
    for y in v:
        x = x + Variable(y[0])[0].detach().numpy()
    x /= (len(v) +len(t))
    pickle.dump(x, open( SET_D + CHR+"_avg.p", "wb" ) )

def plotMatrix(directory, fileName):
    name = os.path.splitext(fileName)[0]
    args = ["--matrix",directory + fileName, "-out", IMAGE_D+name+".png",
            "--log1p", "--dpi", "300"]
    hicPlot.main(args)


def flattenMatrix(matrix):
    flattened = []
    matrix = np.asarray(matrix)
    width = len(matrix)
    for i in range(width):
        tmp = i + CUT_W
        if width < tmp:
            cut = width
        else:
            cut = tmp
        cutM = matrix[i:i+1,i:cut].flatten().tolist()
        if tmp > width:
            cutM.extend(np.zeros(tmp-width))
    flattened.append(cutM)
    return flattened

def putFlattenedHorizontal(flattened):
    matrix = np.zeros((LENGTH,LENGTH))
    e = 0
    print(len(flattened), len(flattened[0]))
    for i in range(LENGTH):
        for j in range(CUT_W):
        #print(flattened[i])
            matrix[i][j] = flattened[i][j]
            matrix[j][i] = flattened[i][j]
    #print(matrix)
    return(matrix)


def reverseFlattenedMatrix(flattened):
    matrix = np.zeros((LENGTH,LENGTH))
    e = 0
    for i in range(LENGTH):
        x = 0
        tmp = i + CUT_W
        if LENGTH < tmp:
            cut = LENGTH
            x = tmp -LENGTH 
        else:
            cut = tmp
        a = CUT_W - x
        matrix[i:i+1, i:cut] = flattened[i][:a]
    return(matrix)


def cutMatrix(ma,chromosome, cutLength =200,  overlap = 50):
    matrix = ma.matrix
    matrix = matrix.todense()
    matrix = np.triu(matrix)
    matrix = np.tril(matrix, CUT_W-1)
    done = False
    length, width = matrix.shape
    start = 0
    end = cutLength
    cuts = ma.cut_intervals
    while(not done):
        if(end >= length):
            end = length - 1
            start = end - cutLength
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
        region = chromosome+":"+str(first)+"-"+str(last)
        m = hm.hiCMatrix(None)
        m.setMatrix(chunk,corrCuts)
        region_end =  int(int(region.split("-")[1]) / 10000)
        name = region.split(":")[0] +"_" + str(region_end).zfill(5) 
        m.save(CHUNK_D+name+".cool")
        start += cutLength - overlap
        end += cutLength - overlap

def iterateAll():
    allChunks = []
    for f in os.listdir(CHROM_D):
        ma = hm.hiCMatrix(CHROM_D+f)
        print(f, ma.matrix.shape)
        cutMatrix(ma, ma.getChrNames()[0])

def printAll(chromosome):
    d = "../Data/Chunks200/"
    for f in os.listdir(d):
        c = f.split("_")[0]
        if chromosome == int(c):
                plotMatrix(d,f)
                ma = hm.hiCMatrix(d+f)

def createDataset(create_test =False):
    a = pickle.load( open( SET_D+CHR+"_avg.p", "rb" ) )
    if create_test:
        d = TEST_D
    else:
        d = CHUNK_D
    matrixList = []
    for f in os.listdir(d):
        c = f.split("_")[0]
        if CHR == c:
            print(f)
            ma = hm.hiCMatrix(d+f)
            matrix = ma.matrix.todense()
            if AVG:
                matrix -= a
            matrix = np.triu(matrix)
            safe = deepcopy(matrix)

            maxV = matrix.max()
            if np.isnan(matrix).any() or maxV == 0:
                continue
            if LOG:
                matrix += 1
                matrix = np.log(matrix) /np.log( MAX)
            elif DIVIDE:
                matrix  = matrix/ MAX 
            if DIAG:
                matrix = flattenMatrix(matrix) 
            else: 
                matrix = np.asarray(matrix)
            #r = reverseFlattenedMatrix(matrix)

            #r *= np.log(MAX)
            #r = np.exp(r)
            #r -= 1
            #print(np.allclose(r,safe))
            #print(safe[-1][-20:])
            #print(r[-1][-20:])
            matrixList.append(matrix)
    else:
        tensor_x = torch.cat(tuple(torch.Tensor([[i]]) for i in matrixList))
    print(tensor_x.shape)
    dataset = utils.TensorDataset(tensor_x)
    div = ""
    diag = ""
    log = ""
    test = ""
    avg = ""
    if AVG:
        avg = "_avg"
    if LOG:
        log = "_log"
    if DIAG:
        diag = "_diag"
    if DIVIDE:
        div = "_div"
    if create_test:
        test = "_test"
    pickle.dump(dataset, open( SET_D + CHR+log+div+diag+avg+test+".p", "wb" ) )


def plotMatrix2(chunk):
        matrix = chunk[1]
        region = chunk[0]
        #mask
        mask = matrix == 0
        mask_nan = np.isnan(matrix)
        mask_inf = np.isinf(matrix)
        log.debug("any nan {}".format(np.isnan(matrix).any()))
        log.debug("any inf {}".format(np.isinf(matrix).any()))

        try:
            matrix[mask] = np.nanmin(matrix[mask == False])
            matrix[mask_nan] = np.nanmin(matrix[mask_nan == False])
            matrix[mask_inf] = np.nanmin(matrix[mask_inf == False])

        except Exception:
            log.debug("Clearing of matrix failed.")
        log.debug("any nanafter remove of nan: {}".format(np.isnan(matrix).any()))
        log.debug("any inf after remove of inf: {}".format(np.isinf(matrix).any()))
       #LOG
        matrix += 1
        norm = LogNorm()
        #SIZES
        fig_height = 7
        height = 4.8 / fig_height
        fig_width = 8
        width = 5.0 / fig_width
        left_margin = (1.0 - width) * 0.5
        bottom = 1.3 / fig_height
        position = [left_margin, bottom, width, height]
        print(region)
        args = ["--matrix", "","-out","../Images/"+region+".png","--title",
                region,"--region", region, "--log1p"]
        args = hicPlot.parse_arguments().parse_args(args)
        cmap = cm.get_cmap(args.colorMap)
        log.debug("Nan values set to black\n")
        cmap.set_bad('black')

        fig = plt.figure(figsize=(fig_width, fig_height))
        hicPlot.plotHeatmap(matrix, None, fig, position,
                    args, cmap, pNorm=norm)

def applyAE(modelName='autoencoder.pt', test=False): 
    model =SparseAutoencoder()
    model.load_state_dict(torch.load(MODEL_D+modelName))
    model.eval()
    if test:
        d = TEST_D
        end = "11000"
    else:
        d = CHUNK_D 
        end = "15200"
    name = CHR + "_"+end
    matrix = name +".cool"
    ma = hm.hiCMatrix(d+matrix)
    m = ma.matrix.todense()
    real_m = ma.matrix.todense()
    a = pickle.load( open( SET_D+CHR+"_avg.p", "rb" ) )
    if AVG:
        m -= a
    if LOG:
        m += 1
        m = np.log(m) /  np.log(MAX)
    elif DIVIDE:
        m = m / MAX 
        length = len(m)
    if  DIAG:
        m = flattenMatrix(m) 
    else:
        m = m.tolist()
    t = torch.Tensor([[m]])
    encoded, decoded = model(t)
    decoded = decoded[0][0]
    # print(decoded.shape)
    # print(m[-1])
    # print(decoded[-1][:100])
    new = decoded.detach().numpy()
    if LOG:
        new *= np.log(MAX)
        new = np.exp(new)
        new -= 1
    if DIVIDE:
        new *= MAX
    if DIAG:
        new = reverseFlattenedMatrix(new)
    if AVG:
        new += a
    new = sparse.csr_matrix(new)
    #plotMatrix(d,matrix)
    ma.setMatrix(new, ma.cut_intervals)
    ma.save(PRED_D + name + "_P_L1.cool")
    plotMatrix(PRED_D, name + "_P_L1.cool")
    


    # real_m = sparse.csr_matrix(real_m - a)
    # ma.setMatrix(real_m, ma.cut_intervals)
    # ma.save(PRED_D + name + "_sub_avg.cool")
    # plotMatrix(PRED_D, name + "_sub_avg.cool")
    # a = sparse.csr_matrix(a)
    # ma.setMatrix(a, ma.cut_intervals)
    # ma.save(PRED_D + CHR+ "_avg.cool")
    # plotMatrix(PRED_D, CHR + "_avg.cool")

def showDiagonal():
    d = TEST_D
    start = "15"
    chrom = "4"
    name = chrom + "_"+start
    matrix = name +".cool"
    ma = hm.hiCMatrix(d+matrix)
    m = ma.matrix.todense()
    m += 1
    m = np.log(m) /  np.log(MAX)
    m = np.asarray(m)
    m = flattenMatrix(m) 
    for i in range(5):
        for j in range(5):
            m[i+25][j+5]= 1
    m2 = reverseFlattenedMatrix(m) 
    m = putFlattenedHorizontal(m)
    m = sparse.csr_matrix(m)
    ma.setMatrix(m, ma.cut_intervals)
    pDir = PRED_D
    ma.save(PRED_D + name + "_Weird.cool")
    plotMatrix(PRED_D, name + "_Weird.cool")
    
    m2 = sparse.csr_matrix(m2)
    ma.setMatrix(m2, ma.cut_intervals)
    ma.save(PRED_D + name + "_WeirdC.cool")
    plotMatrix(PRED_D, name + "_WeirdC.cool")



def main(args=None):

    parser = argparse.ArgumentParser(description='HiC Prediction')

    parserRequired = parser.add_argument_group('Required arguments')

    # define the arguments
    parserRequired.add_argument('--action', '-a', choices=['train', 'create',
         'predict','cutMatrix', 'createAverage'], help='Action to take', required=True)

    parserOpt = parser.add_argument_group('Optional arguments')
    parserOpt.add_argument('--learningRate', '-lr',type=float, default=0.0001)
    parserOpt.add_argument('--epochs', '-e',type=int, default=200)
    parserOpt.add_argument('--chrom', '-c',type=int, default=4)
    parserOpt.add_argument('--cutLength', '-cl',type=int, default=200)
    parserOpt.add_argument('--cutWidth', '-cw',type=int, default=200)
    parserOpt.add_argument('--maxValue', '-mv',type=int, default=4846)
    parserOpt.add_argument('--batchSize', '-b',type=int, default=132)
    parserOpt.add_argument('--model', '-m',type=str, default='autoencoder.pt')
    parserOpt.add_argument('--mse', '-mse', default=False)
    parserOpt.add_argument('--validationData', '-v',default=False)
    parserOpt.add_argument('--trainData', '-t',default=True)
    print(args)
    args = parser.parse_args(args)
    print(args)
    if args.action == "train":
        LEARN_R = args.learningRate
        CHROM = args.chrom
        N_EPOCHS = args.epochs
        BATCH_SIZE = args.batchSize
        mse = args.mse 
        model = args.model
        train(mse=mse, modelName=model)
    elif args.action == "create":
        CHROM = args.chrom
        MAX =args.maxValue
        if args.validationData:
            createDataset(True)
        if args.trainData:
            createDataset(False)
    elif args.action == "predict":
        CHROM = args.chrom
        MAX =args.maxValue
        model = args.model
        if args.validationData:
            applyAE(modelName=model, test=True)
        if args.trainData:
            applyAE(modelName=model, test=False)
    elif args.action == "cutMatrix":
        CHROM = args.chrom
        MAX =args.maxValue
        CUT_W = args.cutWidth
        LENGTH = args.cutLength
        iterateAll()
    elif args.action == "createAverage":
        CHROM = args.chrom
        MAX =args.maxValue
        createAverage()


if __name__ == "__main__":
   torch.set_num_threads(5)
   main(sys.argv[1:])
   print(torch.get_num_threads() )
