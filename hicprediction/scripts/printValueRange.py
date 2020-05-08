import click
from hicmatrix import HiCMatrix as hm
import math
import numpy as np

@click.option('-m','--matrix',type=click.Path(exists=True,dir_okay=False,readable=True),help="HiC-Matrix in cooler format")
@click.option('-lb','--lowerBound', type=click.IntRange(min=0), help="lower bound")
@click.option('-ub', '--upperBound', type=click.IntRange(min=0), help="upper bound")
@click.option('-chr', '--chromosome', type=str, help="chromosome to get values from, depending on format e.g. 'chr1' or just '1'")
@click.command()

def printValueRange(matrix, lowerbound, upperbound, chromosome):
    try:
        hiCMatrix = hm.hiCMatrix(matrix, pChrnameList=[chromosome])
        fullReadsCsr = hiCMatrix.matrix
        resolutionInt = hiCMatrix.getBinSize()
    except:
        msg = "Could not load matrix {:s}, maybe no cooler matrix or chromosome doesn't exist?"
        msg = msg.format(matrix)
        raise SystemExit(msg)

    print("Successfully read matrix {:s}".format(matrix))
    print("bin size (resolution) is {:d}".format(resolutionInt))

    try:
        lowerBin, upperBin = hiCMatrix.getRegionBinRange(chromosome, lowerbound, upperbound)
        readMatrixCsr = fullReadsCsr[lowerBin:upperBin,lowerBin:upperBin]
    except:
        msg = "invalid bounds, upper bound too large? Allowed values:\n"
        msg += "chrom   upper bound\n"
        chromSizeDict = hiCMatrix.get_chromosome_sizes()
        for key in chromSizeDict:
            msg += str(key) + " " + str(chromSizeDict[key]) + "\n"
        raise SystemExit(msg)

    rangeMin = readMatrixCsr.min()
    rangeMax = readMatrixCsr.max()
    rangeMean = np.round(readMatrixCsr.mean(),3)

    print("min. value in range:", rangeMin)
    print("max. value in range:", rangeMax)
    print("mean value in range:", rangeMean)

    chromMin = fullReadsCsr.min()
    chromMax = fullReadsCsr.max()
    chromMean = np.round(fullReadsCsr.mean(), 3)

    print("min. value in chrom:", chromMin)
    print("max. value in chrom:", chromMax)
    print("mean value in chrom:", chromMean)

if __name__ == "__main__":
    printValueRange() # pylint: disable=no-value-for-parameter 