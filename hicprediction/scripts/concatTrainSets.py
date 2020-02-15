import click
import os
import pandas as pd
import joblib
import sys
import numpy as np


@click.argument("datasetpaths", nargs=-1, type=click.Path(exists=True))
@click.option("--ignorechrom", "-igc", type=bool, default=False, help="allow concatenating datasets from different chromosomes")
@click.option("--ignoremerge", "-igm", type=bool, default=False, help="allow concatenating datasets with different merge operations")
@click.option("--ignorewindowop", "-igw", type=bool, default=False, help="allow concatenating datasets with different window operations")
@click.option("--outfile", "-o", type=click.Path(writable=True), help="path to outfile, must have .z file extension", required=True)
@click.command()

def concatTrainSets(datasetpaths, ignorechrom, ignoremerge, ignorewindowop, outfile):
    if not datasetpaths or len(datasetpaths) == 1:
        print("no datasets to concat, exiting")
        return
    else:
        try:
            allTheData = [joblib.load(datasetfile) for datasetfile in datasetpaths]
        except:
            msg = "one of the datasets " + ", ".join(datasetpaths) +  " is not a proper dataset"
            sys.exit(msg)
        datasets = [row[0] for row in allTheData]
        params = [row[1] for row in allTheData]
        if not outfile.endswith(".z"):
            outfilename = os.path.splitext(outfile)[0]
            outfile = outfilename + ".z"
            msg = "outfile name must have .z file extension\n"
            msg += "renamed outfile, now writing to {0:s}"
            print(msg.format(outfile))
        #check if the datasets can be merged: 
        #column names should agree, data should be from same chromosome, same resolution etc.
        #unless specified otherwise via --ignoreX options
        datasetMismatch = False
        errorMessage = "Error:\n"

        #check if the datasets have the same number of columns
        #this happens if they have been created with a different number of proteins
        #for now, just assume same number of columns => same number of proteins => can't allow different column numbers
        nrColumnsSet = set ( str(dataset.shape[1]) for dataset in datasets )
        if len(nrColumnsSet) > 1:
            errorMessage += "Datasets have different number of columns (proteins).\n"
            errorMessage += "Numbers of columns: " + ", ".join(nrColumnsSet) + "\n"
            errorMessage += "Cannot concatenate such sets\n"
            datasetMismatch = True


        #check the chromosomes to which the datasets originally belonged
        #concat is possbile and allowed, if all belong to the same chromosome or --ignorechrom is set
        chromosomeSet = set( np.hstack( [str(param["chrom"]) for param in params] ) )
        if len(chromosomeSet) > 1 and not ignorechrom:
            diffChromStr = ", ".join(chromosomeSet)
            errorMessage += "Datasets are from different chromosomes " + diffChromStr + "\n"
            errorMessage += "flag --ignorechrom True must be used to concatenate such datasets\n"
            datasetMismatch = True

        #check if the datasets have the same resolution
        resolutionSet = set ( np.hstack( [str(param["resolution"]) for param in params] ) )
        if len(resolutionSet) > 1:
            diffResStr = ", ".join(resolutionSet)
            errorMessage += "Datasets have different resolutions " + diffResStr + " bp\n"
            errorMessage += "Cannot concatenate such sets\n"
            datasetMismatch = True

        #check if the datasets have the same window size
        #might make sense to allow different sizes for different cell lines
        #for now, just regard it as an error, if they are not all equal
        windowSizeSet = set ( np.hstack( [str(param["windowSize"]) for param in params] ) )
        if len(windowSizeSet) > 1:
            diffWinSizeStr = ", ".join(windowSizeSet)
            errorMessage += "Datasets have different window sizes " + diffWinSizeStr + "vbp\n"
            errorMessage += "Cannot concatenate such sets\n"
            datasetMismatch = True

        #check if the datasets have the same merge operation
        #merge is possible and allowed, if they are all equal or --ignoremerge is set
        mergeOpSet = set( np.hstack( [str(param["mergeOperation"]) for param in params] ) )
        if len(mergeOpSet) > 1 and not ignoremerge:
            diffMergeOpStr = ", ".join(mergeOpSet)
            errorMessage += "Datasets have different merge operations " + diffMergeOpStr + "\n"
            errorMessage += "flag --ignoremerge True must be used to concat such datasets\n"
            datasetMismatch = True
        
        #check if the datasets have the same window operation
        #merge is possible and allowed, if they are all equal or --ignorewindowop is set
        windowOpSet = set( np.hstack( [str(param["windowOperation"]) for param in params] ) )
        if len(windowOpSet) > 1 and not ignorewindowop:
            diffWindowOpStr = ", ".join(windowOpSet)
            errorMessage += "Datasets have different window operations " + diffWindowOpStr + "\n"
            errorMessage += "flag --ignorewindowop True must be used to concat such datasets\n"
            datasetMismatch = True

        #check if the datasets are either all normalized or all not normalized
        normSet = set( np.hstack( [param["normalize"] for param in params] ) )
        if len(normSet) > 1:
            errorMessage += "Some datasets are normalized and some are not\n"
            errorMessage += "Cannot concatenate such sets\n"
            datasetMismatch = True

        #check if the datasets are all from the same cell line
        #this is allowed, but may not be intended, so issue a warning
        cellLineSet = set( np.hstack( [str(param["cellType"]) for param in params] ) )
        if len(cellLineSet) != len(datasets):
            warnMsg = "Warning: at least two datasets are from the same cell line"
            print(warnMsg)

        #write the dataset
        #if there are multiple parameters (e. g. cell types), write a list instead of a scalar
        if not datasetMismatch:
            paramsDict = params[0]
            if len(cellLineSet) > 1:
                paramsDict["cellType"] = sorted(list(cellLineSet))
            if len(chromosomeSet) > 1:
                paramsDict["chrom"] = sorted(list(chromosomeSet))
            if len(mergeOpSet) > 1:
                paramsDict["mergeOperation"] = sorted(list(mergeOpSet))
            if len(windowOpSet) > 1:
                paramsDict["windowOperation"] = sorted(list(windowOpSet))
            concatDataset = pd.concat(datasets, ignore_index=True, sort=False)
            if not concatDataset.shape[1] == datasets[0].shape[1]:
                #occurs only if columns had different names, since their original shapes are matching
                msg = "Columns of datasets have correct number of features but different feature names\n."
                msg += "Cannot concatenate such sets"
                sys.exit(msg)
            else:
            joblib.dump((concatDataset, paramsDict), outfile, compress=True)
        else:
            sys.exit(errorMessage)

if __name__ == '__main__':
    concatTrainSets()