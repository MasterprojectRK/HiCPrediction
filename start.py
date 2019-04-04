from hicmatrix import HiCMatrix as hm
from hicexplorer import hicPlotMatrix as hicPlot
import numpy as np
from scipy import sparse
import os
import logging as log
import sys
from pprint import pprint
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from Autoencoders_Variants import sparse_autoencoder_l1 as SAEL1
from Autoencoders_Variants import data_utils as du
import torch
import torch.utils.data as utils
import pickle
from sparse_AE import SparseAutoencoder
log.basicConfig(level=log.DEBUG)
np.set_printoptions(threshold=sys.maxsize)



CUT_W =20

def plotMatrix(directory, fileName):
    name = os.path.splitext(fileName)[0]
    args = ["--matrix",directory + fileName, "-out", "../Images/"+name+".png",
            "--log1p","--clearMaskedBins", "--dpi", "300"]
    hicPlot.main(args)


def convertBigMatrix(matrixFile):
    hic_ma = hm.hiCMatrix(matrixiFile)
    firstCut = hic_ma.cut_intervals[0][0]
    i = 0
    newCuts = []
    newCuts.append([])
    for cut in hic_ma.cut_intervals:
        if cut[0] == firstCut:
            newCuts[i].append(cut)
        else:
            i+=1
            newCuts.append([])
            newCuts[i].append(cut)
            firstCut = cut[0]
    i = 0
    for key in hic_ma.chrBinBoundaries:
        ma = hm.hiCMatrix(None)
        pair =  hic_ma.chrBinBoundaries[key]
        b1 = pair[0]
        b2 = pair[1]
        newM = hic_ma.matrix[b1:b2,b1:b2]
        ma.setMatrix(newM,newCuts[i])
        ma.save("../Data/Chroms/Chr" + key +"_100kb.cool")
        i += 1


def flattenMatrix(matrix):
    flattened = []
    matrix = np.asarray(matrix)
    
    width = len(matrix)
    for i in range(width):
        tmp = i + CUT_W
        cut = min(width, tmp) 
        cutM = matrix[i:i+1,i:cut].flatten()
        flattened.extend(cutM)
    return flattened


def reverseFlattenedMatrix(flattened, width):
    matrix = np.zeros((width,width))
    e = 0
    for i in range(width):
        x = 0
        tmp = i + CUT_W
        if width <= tmp:
            cut = width
            x = tmp - width
        else:
            cut = tmp
        s = e
        e = s + CUT_W - x
        matrix[i:i+1, i:cut] = flattened[s:e]
    return(matrix)


def cutMatrix(ma,chromosome, cutLength =200,  overlap = 50):
    matrix = ma.matrix
    binSize = 100000
    matrix = matrix.todense()
    matrix = np.triu(matrix)
    matrix = np.tril(matrix, CUT_W)
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
        m.save("../Data/Chunks100kb/"+region.replace(":", "_").replace("-","_")+"_100kb.cool")
        start += cutLength - overlap
        end += cutLength - overlap

def iterateAll():
    d = "../Data/Chroms/"
    allChunks = []
    for f in os.listdir(d):
        ma = hm.hiCMatrix(d+f)
        print(f, ma.matrix.shape)
        cutMatrix(ma, ma.getChrNames()[0])

def printAll(chromosome):
    d = "../Data/Chunks100kb/"
    for f in os.listdir(d):
        c = f.split("_")[0]
        if c is not  "X":
            if chromosome == int(c):
                plotMatrix(d,f)
                ma = hm.hiCMatrix(d+f)

def createDataset():
    d = "../Data/Chunks100kb/"
    matrixList = []
    for f in os.listdir(d):
        c = f.split("_")[0]
        if c is not  "X":
            if 4 == int(c):
                ma = hm.hiCMatrix(d+f)
                matrix = ma.matrix.todense()
                matrix = matrix / 55994
                flattened = flattenMatrix(matrix) 
                print(len(flattened))
                matrixList.append(flattened)
    tensor_x = torch.stack([torch.Tensor(i) for i in matrixList])
    dataset = utils.TensorDataset(tensor_x)
    #print(dataset[0])
    pickle.dump(dataset, open( "../Data/chr4_100kb.p", "wb" ) )
 


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
                region,"--region", region, "--log1p","--clearMaskedBins"]
        args = hicPlot.parse_arguments().parse_args(args)
        cmap = cm.get_cmap(args.colorMap)
        log.debug("Nan values set to black\n")
        cmap.set_bad('black')

        fig = plt.figure(figsize=(fig_width, fig_height))
        hicPlot.plotHeatmap(matrix, None, fig, position,
                    args, cmap, pNorm=norm)

def applyAE():
    
    model =SparseAutoencoder(3810,1000)
    model.load_state_dict(torch.load("../Data/autoencoder.pt"))
    model.eval()
    matrix = "../Data/Chunks100kb/4_15000000_35000000_100kb.cool"
    ma = hm.hiCMatrix(matrix)
    m = ma.matrix.todense()
    m = m /  55994
    length = len(m)
    m = flattenMatrix(m) 
    print(m[-10:])
    print(len(m))
    t = torch.Tensor(m)
    encoded, decoded = model(t)
    #print(m)
    print(decoded)
    new = decoded.detach().numpy()
    print(new[-10:])
    new *= 55994
    new = reverseFlattenedMatrix(new,length)
    new = sparse.csr_matrix(new)
    plotMatrix("../Data/Chunks100kb/","4_15000000_35000000_100kb.cool")
    ma.setMatrix(new, ma.cut_intervals)
    ma.save("../Data/Chunks100kbPredicted/4_15_P.cool")
    plotMatrix("../Data/Chunks100kbPredicted/", "4_15_P.cool")



#matrix = "../Data/GSE63525_GM12878_insitu_primary_100kb.cool"
#matrix = "../Data/GSE63525_GM12878_insitu_primary_100kb_KR_chr1.cool"
# ae = SAEL1.SparseAutoencoderL1()
#matrix = "../Data/Chroms/ChrY_100kb.cool"
#printMatrix(matrix, "ChrY")
#matrix = "../Data/Chroms/Chr2_100kb.cool"
#printMatrix(matrix, "Chr2")
#matrix = "../Data/Chr3_100kb.cool"
#printMatrix(matrix, "Chr3")
#matrix = "../Data/Chr4_100kb.cool"
#printMatrix(matrix, "Chr4")
#iterateAll()
#printAll(5)
#plotMatrix("../Data/Chroms/","Chr5_100kb.cool")
#createDataset()
applyAE()

