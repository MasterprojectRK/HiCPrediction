import click
import cooler
from hicmatrix import HiCMatrix as hm
import pandas as pd
from hicprediction.createTrainingSet import maskFunc
import numpy as np
from hicprediction.predict import getCorrelation
from hicprediction.tagCreator import getResultFileColumnNames
import sklearn.metrics as metrics

@click.option('-i1','--infile1',required=True,
                type=click.Path(exists=True,dir_okay=False,readable=True),
                help="Cooler matrix 1, 'Prediction', should be adjusted to only one chromosome")
@click.option('-i2','--infile2',required=True,
                type=click.Path(exists=True,dir_okay=False,readable=True),
                help="Cooler matrix 2, 'Target', should be adjusted to only one chromsome")
@click.option('-ws','--windowsize', type=click.IntRange(min=1), default=1e6, 
                required=False, help="window size in basepairs")
@click.option('-o','--outfile',required=True,
                type=click.Path(file_okay=True,writable=True),
                help="path and filename for outfile")
@click.option('-pct','--predictionCellType',type=str, default="unknown", help="Cell type for prediction")
@click.option('-mct','--modelCellType', type=str, default="unknown", help="Cell type(s) of model")
@click.option('-mchr', '--modelChromosome', type=str, default="unknown", help="Chromosome used for training, e.g. chr1")
@click.command()
def computeMetrics(infile1,infile2,windowsize,outfile, predictioncelltype, modelcelltype, modelchromosome):
    #try loading HiC matrices
    try:
        hicMatrix1 = hm.hiCMatrix(infile1)
        hicMatrix2 = hm.hiCMatrix(infile2)
    except Exception as e:
        print(e)
        msg = "Could not load matrices, probably no cooler format"
        raise SystemExit(msg)
    
    #check bin sizes, must be equal / same matrix resolution
    binSize1 = hicMatrix1.getBinSize()
    binSize2 = hicMatrix2.getBinSize()
    if binSize1 != binSize2:
        msg = "Aborting. Bin sizes not equal.\n"
        msg += "Bin size 1: {0:d}, bin size 2: {0:d}"
        msg = msg.format(binSize1, binSize2)
        raise SystemExit(msg)
    numberOfDiagonals = int(np.round(windowsize/binSize1))
    if numberOfDiagonals < 1:
        msg = "Window size must be larger than bin size of matrices.\n"
        msg += "Remember to specify window in basepairs, not bins."
        raise SystemExit(msg)

    #check chromosomes
    chromList1 = hicMatrix1.getChrNames()
    chromList2 = hicMatrix2.getChrNames()
    if chromList1 and chromList2:
        chrom1Str = str(chromList1[0])
        chrom2Str = str(chromList2[0])
        if chrom1Str != chrom2Str:
            msg = "Aborting, chromosomes are not the same: {:s} vs. {:s}"
            msg = msg.format(chrom1Str, chrom2Str)
            raise SystemExit(msg)
        if len(chromList1) != 1 or len(chromList2) != 1:
            msg = "Warning, more than one chromosome in the matrix\n"
            msg += "Consider using e.g. hicAdjustMatrix with --keep on the desired chromosome.\n"
            msg += "Only taking the first chrom, {:s}"
            msg = msg.format(chrom1Str)
    else:
        msg = "Aborting, no chromosomes found in matrix"
        raise SystemExit(msg)

    sparseMatrix1 = hicMatrix1.matrix
    sparseMatrix2 = hicMatrix2.matrix
    shape1 = sparseMatrix1.shape
    shape2 = sparseMatrix2.shape
    if shape1 != shape2:
        msg = "Aborting. Shapes of matrices are not equal.\n"
        msg += "Shape 1: ({:d},{:d}); Shape 2: ({:d},{:d})"
        msg = msg.format(shape1[0],shape1[1],shape2[0],shape2[1])
        raise SystemExit(msg)
    if numberOfDiagonals > shape1[0]-1:
        msg = "Aborting. Window size {0:d} larger than matrix size {:d}"
        msg = msg.format(numberOfDiagonals, shape1[0]-1)
        raise SystemExit(msg)    

    trapezIndices = np.mask_indices(shape1[0],maskFunc,k=numberOfDiagonals)
    reads1 = np.array(sparseMatrix1[trapezIndices])[0]
    reads2 = np.array(sparseMatrix2[trapezIndices])[0]

    matrixDf = pd.DataFrame(columns=['first','second','distance','reads1','reads2'])
    matrixDf['first'] = np.uint32(trapezIndices[0])
    matrixDf['second'] = np.uint32(trapezIndices[1])
    matrixDf['distance'] = np.uint32(matrixDf['second'] - matrixDf['first'])
    matrixDf['reads1'] = np.float32(reads1)
    matrixDf['reads2'] = np.float32(reads2)
    matrixDf.fillna(0, inplace=True)

    pearsonAucIndices, pearsonAucValues = getCorrelation(matrixDf,'distance', 'reads1', 'reads2', 'pearson')
    pearsonAucScore = metrics.auc(pearsonAucIndices, pearsonAucValues)
    spearmanAucIncides, spearmanAucValues = getCorrelation(matrixDf,'distance', 'reads1', 'reads2', 'spearman')
    spearmanAucScore = metrics.auc(spearmanAucIncides, spearmanAucValues)
    corrScoreOPredicted_Pearson = matrixDf[['reads1','reads2']].corr(method= \
                'pearson').iloc[0::2,-1].values[0]
    corrScoreOPredicted_Spearman= matrixDf[['reads1', 'reads2']].corr(method= \
                'spearman').iloc[0::2,-1].values[0]
    print("PearsonAUC", pearsonAucScore)
    print("SpearmanAUC", spearmanAucScore)

    columns = getResultFileColumnNames(sorted(list(matrixDf.distance.unique())))
    resultsDf = pd.DataFrame(columns=columns)
    resultsDf.set_index('Tag', inplace=True)
    tag = 'xxx'
    resultsDf.loc[tag, 'R2'] = metrics.r2_score(matrixDf['reads2'], matrixDf['reads1'])
    resultsDf.loc[tag, 'MSE'] = metrics.mean_squared_error( matrixDf['reads2'], matrixDf['reads1'])
    resultsDf.loc[tag, 'MAE'] = metrics.mean_absolute_error( matrixDf['reads2'], matrixDf['reads1'])
    resultsDf.loc[tag, 'MSLE'] = metrics.mean_squared_log_error(matrixDf['reads2'], matrixDf['reads1'])
    resultsDf.loc[tag, 'AUC_OP_P'] = pearsonAucScore 
    resultsDf.loc[tag, 'AUC_OP_S'] = spearmanAucScore
    resultsDf.loc[tag, 'S_OP'] = corrScoreOPredicted_Spearman 
    resultsDf.loc[tag, 'P_OP'] = corrScoreOPredicted_Pearson
    resultsDf.loc[tag, 'resolution'] = binSize1
    resultsDf.loc[tag, 'modelChromosome'] = modelchromosome
    resultsDf.loc[tag, 'modelCellType'] = modelcelltype
    resultsDf.loc[tag, 'predictionChromosome'] = chrom1Str 
    resultsDf.loc[tag, 'predictionCellType'] = predictioncelltype
    for i, pearsonIndex in enumerate(pearsonAucIndices):
        columnName = int(round(pearsonIndex * matrixDf.distance.max()))
        resultsDf.loc[tag, columnName] = pearsonAucValues[i]
    resultsDf = resultsDf.sort_values(by=['predictionCellType','predictionChromosome',
                            'modelCellType','modelChromosome', 'conversion',\
                            'Window','Merge', 'normalize'])
    resultsDf.to_csv(outfile)

if __name__ == "__main__":
    computeMetrics()