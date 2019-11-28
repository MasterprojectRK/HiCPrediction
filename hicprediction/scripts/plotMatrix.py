#!/usr/bin/env python3
import os
os.environ['NUMEXPR_MAX_THREADS'] = '16'
os.environ['NUMEXPR_NUM_THREADS'] = '8'
import hicprediction.configurations as conf
from hicmatrix import HiCMatrix as hm
from hicexplorer import hicPlotMatrix as hicPlot
from scipy.sparse import triu, tril
from argparse import Namespace
import numpy as np
import logging as log
log.basicConfig(level=log.DEBUG)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import click
from matplotlib.colors import LogNorm

@click.option('--regionIndex1', '-r1',default=1, show_default=True, required=True)
@click.option('--regionIndex2','-r2', default=1000, show_default=True, required=True)
@click.option('--matrixinputfile', '-mif',type=click.Path(exists=True), required=True)
@click.option('--imageoutputfile','-iof', default=None)
@click.option('--comparematrix', '-cmp', type=click.Path(exists=True), required=True)
@click.option('--title', '-t', type=str, default=None)
@click.command()
def plotMatrix(matrixinputfile,imageoutputfile, regionindex1, regionindex2, comparematrix, title):
        if not imageoutputfile:
            imageoutputfile = matrixinputfile.split('.')[0] +'.png'
        conf.checkExtension(matrixinputfile, 'cool')
        conf.checkExtension(imageoutputfile, 'png')
            
        #get the full matrix first to extract the desired region
        ma = hm.hiCMatrix(comparematrix)
        cuts = ma.cut_intervals
        chromosome = cuts[0][0]
        region = str(chromosome) +":"+str(cuts[regionindex1][1])+"-"+ str(cuts[regionindex2][1])
        
        #now get the predicted and the compared matrix, restricted to the desired region
        lowerHiCMatrix = hm.hiCMatrix(comparematrix , pChrnameList=[region])
        upperHiCMatrix = hm.hiCMatrix(matrixinputfile ,pChrnameList=[region]) #todo: load region from matrix
        #only use upper and lower triangles
        lowerMatrix = tril(lowerHiCMatrix.matrix, k=0, format="csr")
        upperMatrix = triu(upperHiCMatrix.matrix, k=1, format="csr")
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
                             matrix = comparematrix) 
        
        #following code is duplicated from hicPlotMatrix
        #not exactly beautiful, but works for now
        chrom, region_start, region_end, idx1, start_pos1, chrom2, region_start2, region_end2, idx2, start_pos2 = hicPlot.getRegion(plotArgs, lowerHiCMatrix)
        
        mixedMatrix = np.asarray((lowerMatrix + upperMatrix).todense().astype(float))
        
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

# def plotMatrix(args):
    # for i in range(1,4):
        # print(i)
        # args.regionIndex1 = i*500 + 1
        # args.regionIndex2 = (i+1)*500
        # name = args.sourceFile.split(".")[0].split("/")[-1]
        # a = ["--matrix",args.sourceFile,
                # "--dpi", "300"]
        # if args.log:
            # a.extend(["--log1p", "--vMin" ,"1","--vMax" ,"1000"])
        # else:
            # a.extend(["--vMax" ,"1","--vMin" ,"0"])
        # if args.region:
            # a.extend(["--region", args.region])
            # name = name + "_r"+args.region
        # elif args.regionIndex1 and args.regionIndex2:
            # ma = hm.hiCMatrix(args.sourceFile)
            # cuts = ma.cut_intervals
            # region = args.chrom +":"+str(cuts[args.regionIndex1][1])+"-"+ str(cuts[args.regionIndex2][1])
            # a.extend(["--region", region])
            # name = name + "_R"+region

        # a.extend( ["-out", IMAGE_D+name+".png"])
        # hicPlot.main(a)


# def plotDir(args):
    # for cs in [11,14,17,9,19]:
        # args.chroms = str(cs)
        # for c in ["9_A"]:
            # args.chrom = str(c)
            # for p in ["default"]:
                # args.conversion = p
                # for w in ["avg"]:
                    # args.windowOperation = w
                    # for me in ["avg"]:
                        # args.mergeOperation = me
                        # for m in ["rf"]:
                            # args.model = m
                            # for n in [False]:
                                # args.normalizeProteins = n
                                # for n in [False]:
                                    # args.equalizeProteins = n
                                    # if os.path.isfile(tagCreator(args,"pred")):
                                        # for i in range(5):
                                            # args.regionIndex1 = i*500 + 1
                                            # args.regionIndex2 = (i+1)*500
                                            # plotPredMatrix(args)


# def concatResults():
    # sets = []
    # for a in os.listdir(RESULTPART_D):
        # if a.split("_")[0] == "part":
            # if os.path.isfile(RESULTPART_D + a):
                # sets.append(pickle.load(open(RESULTPART_D + a, "rb" ) ))
                # print(len(pickle.load(open(RESULTPART_D + a, "rb" ) )))
    # sets.append(pickle.load(open(RESULT_D+"baseResults.p", "rb" ) ))
    # df_all = pd.concat(sets)
    # df_all = df_all.drop_duplicates()
    # print(len(df_all))
    # # df_all = df_all[~df_all.index.duplicated()]
    # print(df_all[df_all.index.duplicated()])
    # print(len(df_all))

    # pickle.dump(df_all, open(RESULT_D+"baseResults.p", "wb" ) )

# def mergeAndSave():
    # d = RESULT_D
    # now = str(datetime.datetime.now())[:19]
    # now = now.replace(":","_").replace(" ", "")
    # src_dir = d + "baseResults.p"
    # dst_dir = d + "/Old/old"+str(now)+".p"
    # shutil.copy(src_dir,dst_dir)
    # concatResults()

if __name__ == '__main__':
    plotMatrix()
