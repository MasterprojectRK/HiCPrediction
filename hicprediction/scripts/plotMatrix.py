#!/usr/bin/env python3
import os
os.environ['NUMEXPR_MAX_THREADS'] = '16'
os.environ['NUMEXPR_NUM_THREADS'] = '8'
import sys
import hicprediction.configurations as conf
from hicmatrix import HiCMatrix as hm
from hicexplorer import hicPlotMatrix as hicPlot
from scipy.sparse import triu, tril
from scipy.sparse import csr_matrix
from argparse import Namespace
import numpy as np
import logging as log
log.basicConfig(level=log.DEBUG)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import click
from matplotlib.colors import LogNorm

@click.option('--regionIndex1', '-r1',default=1, show_default=True, required=True, type=click.IntRange(min=0) )
@click.option('--regionIndex2','-r2', default=1000, show_default=True, required=True, type=click.IntRange(min=0))
@click.option('--matrixinputfile', '-mif',type=click.Path(exists=True), required=True)
@click.option('--imageoutputfile','-iof', default=None)
@click.option('--comparematrix', '-cmp', type=click.Path(exists=True))
@click.option('--title', '-t', type=str, default=None)
@click.command()
def plotMatrix(matrixinputfile,imageoutputfile, regionindex1, regionindex2, comparematrix, title):
        if not conf.checkExtension(matrixinputfile, '.cool'):
            msg = "input matrix must be in cooler format (.cool)"
            sys.exit(msg)
        if comparematrix and not conf.checkExtension(comparematrix, ".cool"):
            msg = "if specified, compare matrix must be in cooler format (.cool)"
            sys.exit(msg)
        if not imageoutputfile:
            imageoutputfile = matrixinputfile.rstrip('cool') + 'png'
        elif imageoutputfile and not conf.checkExtension(imageoutputfile, ".png"):
            imageoutputfile = os.path.splitext(imageoutputfile)[0] + ".png"
       
        #get the full matrix first to extract the desired region
        ma = hm.hiCMatrix(matrixinputfile)
        cuts = ma.cut_intervals
        chromosome = cuts[0][0]
        maxIndex = len(cuts) - 1
        #check indices and get the region if ok
        if regionindex1 > maxIndex:
            msg = "invalid start region. Allowed is 0 to {0:d} (0 to {1:d})".format(maxIndex, cuts[maxIndex][1])
            sys.exit(msg)
        if regionindex2 < regionindex1:
           msg = "region index 2 must be smaller than region index 1"
           sys.exit(msg)
        if regionindex2 > maxIndex:
            regionindex2 = maxIndex
            print("region index 2 clamped to max. value {0:d}".format(maxIndex))
        region = str(chromosome) +":"+str(cuts[regionindex1][1])+"-"+ str(cuts[regionindex2][1])
        
        #now get the data for the input matrix, restricted to the desired region
        upperHiCMatrix = hm.hiCMatrix(matrixinputfile ,pChrnameList=[region])
        upperMatrix = triu(upperHiCMatrix.matrix, k=1, format="csr")
        
        #if set, get data from the same region also for the compare matrix
        #there's no compatibility check so far
        lowerHiCMatrix = None
        lowerMatrix = None
        if comparematrix:
            lowerHiCMatrix = hm.hiCMatrix(comparematrix)
            if chromosome not in [row[0] for row in lowerHiCMatrix.cut_intervals]:
                msg = "compare matrix must contain the same chromosome as the input matrix"
                sys.exit(msg)
            lowerHiCMatrix = hm.hiCMatrix(comparematrix , pChrnameList=[region])
            lowerMatrix = tril(lowerHiCMatrix.matrix, k=0, format="csr") 

            if lowerMatrix.get_shape() != upperMatrix.get_shape():
                msg = "shapes of input matrix and compare matrix do not match. Check resolutions"
                sys.exit(msg)

        #arguments for plotting
        plotArgs = Namespace(bigwig=None, 
                             chromosomeOrder=None, 
                             clearMaskedBins=False, 
                             colorMap='RdYlBu_r', 
                             disable_tight_layout=False, 
                             dpi=300, 
                             flipBigwigSign=False, 
                             log=False, log1p=True, 
                             perChromosome=False, 
                             region=region, 
                             region2=None, 
                             scaleFactorBigwig=1.0, 
                             scoreName=None, 
                             title=title, 
                             vMax=None, vMaxBigwig=None, 
                             vMin=1.0, vMinBigwig=None,
                             matrix = matrixinputfile) 
        
        #following code is largely duplicated from hicPlotMatrix
        #not exactly beautiful, but works for now
        chrom, region_start, region_end, idx1, start_pos1, chrom2, region_start2, region_end2, idx2, start_pos2 = hicPlot.getRegion(plotArgs, upperHiCMatrix)
        

        mixedMatrix = None
        if comparematrix:
            mixedMatrix = np.asarray((lowerMatrix + upperMatrix).todense().astype(float))
        else:
            mixedMatrix = np.asarray(upperHiCMatrix.matrix.todense().astype(float))
        
        #colormap for plotting
        cmap = cm.get_cmap(plotArgs.colorMap)
        cmap.set_bad('black')
        bigwig_info = None

        norm = None

        if plotArgs.log or plotArgs.log1p:
            mask = mixedMatrix == 0
            try:
                mixedMatrix[mask] = np.nanmin(mixedMatrix[mask == False])
            except ValueError:
                log.info('Matrix contains only 0. Set all values to {}'.format(np.finfo(float).tiny))
                mixedMatrix[mask] = np.finfo(float).tiny
            if np.isnan(mixedMatrix).any() or np.isinf(mixedMatrix).any():
                log.debug("any nan {}".format(np.isnan(mixedMatrix).any()))
                log.debug("any inf {}".format(np.isinf(mixedMatrix).any()))
                mask_nan = np.isnan(mixedMatrix)
                mask_inf = np.isinf(mixedMatrix)
                mixedMatrix[mask_nan] = np.nanmin(mixedMatrix[mask_nan == False])
                mixedMatrix[mask_inf] = np.nanmin(mixedMatrix[mask_inf == False])

        log.debug("any nan after remove of nan: {}".format(np.isnan(mixedMatrix).any()))
        log.debug("any inf after remove of inf: {}".format(np.isinf(mixedMatrix).any()))
        if plotArgs.log1p:
            mixedMatrix += 1
            norm = LogNorm()
        elif plotArgs.log:
            norm = LogNorm()

        fig_height = 7
        height = 4.8 / fig_height

        fig_width = 8
        width = 5.0 / fig_width
        left_margin = (1.0 - width) * 0.5

        fig = plt.figure(figsize=(fig_width, fig_height), dpi=plotArgs.dpi)

        ax1 = None
        bottom = 1.3 / fig_height

        position = [left_margin, bottom, width, height]
        hicPlot.plotHeatmap(mixedMatrix, ma.get_chromosome_sizes(), fig, position,
                    plotArgs, cmap, xlabel=chrom, ylabel=chrom2,
                    start_pos=start_pos1, start_pos2=start_pos2, pNorm=norm, pAxis=ax1, pBigwig=bigwig_info)
        plt.savefig(imageoutputfile, dpi=plotArgs.dpi)
        plt.close(fig)

        #the following does not work, unfortunately
        #cooler throws bad input error
        ##lowerHiCMatrix.matrix = lowerMatrix + upperMatrix
        ##lowerHiCMatrix.save("test.cool", pSymmetric=False)

if __name__ == '__main__':
    plotMatrix()
