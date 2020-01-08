import click
import pandas as pd
import sys


@click.option('--trainfile','-tr', help="input tab delimited train data file", required=True, type=click.Path(exists=True))
@click.option('--testfile','-te',help="input tab delimited train data file", required=True, type=click.Path(exists=True))
@click.option('--outDir','-o', help="output directory for the bed files", required=True, type=click.Path(exists=True, writable=True))
@click.command()
def convertHicRegDataToBedgraph(trainfile, testfile, outdir):
    rawDataframe1 = None
    rawDataframe2 = None
    try:
        rawDataframe1 = pd.read_table(trainfile)
        rawDataframe2 = pd.read_table(testfile)
    except:
        msg="could not read all of the infiles, maybe no tab delimited text file(s)?"
        sys.exit(msg)
    rawDataframe = pd.concat([rawDataframe1, rawDataframe2])
    if rawDataframe.shape[1] != rawDataframe1.shape[1] or rawDataframe.shape[1] != rawDataframe2.shape[1]:
        msg = "Different number of columns, maybe different proteins in input files?"
        sys.exit(msg)
    print("rawDF\n", rawDataframe)

    #filter raw data for start values (protein columns that start with  _E)
    startDf = rawDataframe.filter(regex="Pair|_E$").copy(deep=True)
    startDf['Pair'] = startDf['Pair'].apply(lambda x: x.split("-")[0])
    startDf.drop_duplicates(inplace=True)
    startDf['chrom'] = startDf['Pair'].apply(lambda x: x.split("_")[0])
    startDf['chromStart'] = startDf['Pair'].apply(lambda x: int(x.split("_")[1]))
    startDf['chromEnd'] = startDf['Pair'].apply(lambda x: int(x.split("_")[2]))
    startDf.drop(columns=['Pair'], inplace=True)
    startDf.sort_values(by=['chromStart', 'chromEnd'], inplace=True)
    startDf.reset_index(inplace=True, drop=True)
    startDf.columns = [x.replace("_E","") for x in startDf.columns]
    print("startDF\n", startDf)

    #filter raw data for end values (protein columns that start with  _P)
    endDf = rawDataframe.filter(regex="Pair|_P").copy(deep=True)
    endDf['Pair'] = endDf['Pair'].apply(lambda x: x.split("-")[1])
    endDf.drop_duplicates(inplace=True)
    endDf['chrom'] = endDf['Pair'].apply(lambda x: x.split("_")[0])
    endDf['chromStart'] = endDf['Pair'].apply(lambda x: int(x.split("_")[1]))
    endDf['chromEnd'] = endDf['Pair'].apply(lambda x: int(x.split("_")[2]))
    endDf.drop(columns=['Pair'],inplace=True)
    endDf.sort_values(by=['chromStart', 'chromEnd'], inplace=True)
    endDf.reset_index(inplace=True,drop=True)
    endDf.columns = [x.replace("_P","") for x in endDf.columns]
    print("endDf\n", endDf)

    jointDf = None
    if startDf.shape[1] == endDf.shape[1]:
        jointDf = startDf.append(endDf) 
    else:
        msg = "different number of columns ending with _E and _P, check input data"
        sys.exit(msg)
    if jointDf.shape[1] != startDf.shape[1]:
        msg = "error joining _E and _P values. Check input, maybe different Proteins were used?"
        sys.exit(msg)
    jointDf.drop_duplicates(inplace=True) #such duplicates *may* occur for combined train/test set data
    rowsBefore = jointDf.shape[0]
    jointDf.drop_duplicates(subset=['chrom', 'chromStart', 'chromEnd'],inplace=True)
    rowsAfter = jointDf.shape[0]
    if rowsBefore != rowsAfter: #this *must not* happen
        msg = "Aborting. {0:d} regions occur twice. Check input data"
        sys.exit(msg.format(rowsBefore - rowsAfter))
    jointDf.sort_values(by=['chromStart', 'chromEnd'], inplace=True)
    jointDf.reset_index(inplace=True,drop=True)
    print("jointDF\n", jointDf)

    #write bedgraph files for all proteins
    #there are three columns which are not protein data
    proteins = list(jointDf.columns)[0:jointDf.shape[1]-4] 
    for protein in proteins:
        with open(outdir + str(protein) + '.bedGraph', "w") as bedgraphFile: 
            bedgraphFile.write("track type=bedGraph name=" + str(protein) + "\n") 
            jointDf.to_csv(bedgraphFile, 
                        sep='\t', 
                        columns=['chrom','chromStart','chromEnd',protein],
                        header=None,
                        index=False,
                        float_format='%.6f')

if __name__ == '__main__':
    convertHicRegDataToBedgraph()