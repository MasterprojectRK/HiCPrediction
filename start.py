from hicmatrix import HiCMatrix as hm
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
from Autoencoders_Variants import sparse_autoencoder_l1 as SAEL1
from Autoencoders_Variants import data_utils as du
import torch
import torch.utils.data as utils
import pickle
from sparse_AE import DIVIDE
from sparse_AE import DIAG
from sparse_AE import MAX
from sparse_AE import LOG
from sparse_AE import CUT_W
from sparse_AE import SPARSE
from sparse_AE import LENGTH
from sparse_AE import SparseAutoencoder
import cooler
from copy import copy, deepcopy

log.basicConfig(level=log.DEBUG)
np.set_printoptions(threshold=sys.maxsize)

BIN = 10000
DATA_D = "Data2e/"
CHROM_D = DATA_D + "Chroms/"
SET_D = DATA_D + "Sets/"
PRED_D = DATA_D + "Predictions/"
MODEL_D  = DATA_D + "Models/"
CHUNK_D = DATA_D + "Chunks/"
TEST_D =  DATA_D + "Tests/"
IMAGE_D = DATA_D +  "Images/"
ORIG_D = DATA_D +  "Orig/"

def plotMatrix(directory, fileName):
    name = os.path.splitext(fileName)[0]
    args = ["--matrix",directory + fileName, "-out", "../Images/"+name+".png",
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
        m.save(CHUNK_D+region.replace(":",
            "_").split("-")[0].replace("00","")+".cool")
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

def createDataset():
    matrixList = []
    for f in os.listdir(CHUNK_D):
        c = f.split("_")[0]
        if CHR == int(c):
            ma = hm.hiCMatrix(CHUNK_D+f)
            matrix = ma.matrix.todense()
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
                print(len(matrix),len(matrix[0]))
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
    if LOG:
        log = "_log"
    if DIAG:
        diag = "_diag"
    if DIVIDE:
        div = "_div"
    pickle.dump(dataset, open( SET_D + CHR+log+div+diag+".p", "wb" ) )
 


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

def applyAE():
    test = False 
    model =SparseAutoencoder()

    model.load_state_dict(torch.load("../Data/autoencoder.pt"))
    model.eval()
    chrom = "4"
    pDir = "../Data/Chunks100kbPredicted/"
    if test:
        d = "../Data/Test200/"
        start = "15"
    else:
        d = "../Data/Chunks200/"
        start = "0"
    name = chrom + "_"+start
    matrix = name +".cool"
    ma = hm.hiCMatrix(d+matrix)
    m = ma.matrix.todense()
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
    print(decoded.shape)
    print(m[-1])
    print(decoded[-1][:100])
    new = decoded.detach().numpy()
    if LOG:
        new *= np.log(MAX)
        new = np.exp(new)
        new -= 1
    if DIVIDE:
        new *= MAX
    if DIAG:
        new = reverseFlattenedMatrix(new)
    new = sparse.csr_matrix(new)
    plotMatrix(d,matrix)
    ma.setMatrix(new, ma.cut_intervals)
    ma.save(pDir + name + "_P.cool")
    plotMatrix(pDir, name + "_P.cool")

def showDiagonal():
    d = "../Data/Test200/"
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
    pDir = "../Data/Chunks100kbPredicted/"
    ma.save(PRED_D + name + "_Weird.cool")
    plotMatrix(PRED_D, name + "_Weird.cool")
    
    m2 = sparse.csr_matrix(m2)
    ma.setMatrix(m2, ma.cut_intervals)
    ma.save(PRED_D + name + "_WeirdC.cool")
    plotMatrix(PRED_D, name + "_WeirdC.cool")


matrix = "../Data2e/Orig/GSE63525_GM12878_insitu_primary_10kb_KR.cool"
convertBigMatrix(matrix)
#matrix = "../Data/GSE63525_GM12878_insitu_primary_100kb_KR_chr1.cool"
# ae = SAEL1.SparseAutoencoderL1()
#matrix = "../Data/Chroms/ChrY_100kb.cool"
#printMatrix(matrix, "ChrY")
#matrix = "../Data/Chroms/Chr4_Adj.cool"
#plotMatrix("../Data/Chroms/", "Chr4_Adj.cool")
#matrix = "../Data/Chr3_100kb.cool"
#printMatrix(matrix, "Chr3")
#matrix = "../Data/Chr4_100kb.cool"
#printMatrix(matrix, "Chr4")
#iterateAll()
#printAll(4)
#plotMatrix("../Data/Chroms/","Chr5_100kb.cool")
#createDataset()
#applyAE()
#showDiagonal()

