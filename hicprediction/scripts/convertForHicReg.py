import click
import joblib
import pandas as pd
import os

@click.option('--dataset', '-ds', type=click.Path(exists=True, 
                                readable=True,dir_okay=False, 
                                file_okay=True),
                                required=True,
                                help="hicprediction dataset with read counts")
@click.option('--outpath', '-o' , type=click.Path(writable=True, file_okay=False, dir_okay=True),
                                required=True,
                                help="directory for output")
@click.option('--dropInvalid', '-di', type=bool, required=False, default=True)
@click.command()
def convertForHicReg(dataset, outpath, dropinvalid):
   
    #check inputs
    try:
        df, params = joblib.load(dataset)
    except Exception as e:
        msg = "Failed loading dataset {:s}. Wrong format?"
        msg = msg.format(dataset)
        raise SystemExit(msg)
    if not isinstance(df, pd.DataFrame):
        msg = "Aborting. Input {:s} is readable, but not a dataset\n"
        msg = msg.format(dataset)
        raise SystemExit(msg)
    requiredColumns = {'first', 'second', 'chrom', 'reads'}
    if not requiredColumns.issubset(set(df.columns)):
        msg = "Aborting. Dataset {:s} does not have all required columns: "
        msg += ", ".join(requiredColumns)
        msg = msg.format(dataset)
        raise SystemExit(msg)
    resolutionInt = None
    cellLineStr = None
    try:
        resolutionInt = int(params['resolution'])
        cellLineStr = str(params['cellType'])
    except Exception as e:
        msg = "Aborting: Resolution or cell type not in parameter dict / wrong data type"
        raise SystemExit(msg)

    #drop unnecessary columns and use valid samples only, if applicable
    if 'avgRead' in df.columns:
        df.drop('avgRead', axis=1, inplace=True)
    if 'valid' in df.columns:
        if dropinvalid == True:
            df = df[df['valid'] == True]
        df.drop('valid', axis=1, inplace=True)

    #give some feedback
    msg = "Dataset is from cell line {:s}, resolution {:d}.\n"
    msg += "Available columns: " + ", ".join(df.columns)
    msg = msg.format(cellLineStr, resolutionInt)
    print(msg)
    msg = "The dataset contains {:d} samples"
    msg = msg.format(df.shape[0])
    print(msg)

    #build up the dataframes for HiC-Reg
    outDf = pd.DataFrame()
    priorDf = pd.DataFrame(columns=['name', 'count'])

    #build the "Count" column
    outDf['pair1'] = 'chr' + df['chrom'].astype('str')
    outDf['pair2'] = 'chr' + df['chrom'].astype('str')
    outDf['pair1Start'] = df['first'] * resolutionInt
    outDf['pair1End'] = df['first'] * resolutionInt + resolutionInt
    outDf['pair2Start'] = df['second'] * resolutionInt
    outDf['pair2End'] = df['second'] * resolutionInt + resolutionInt
    outDf['pair1'] += '_' + outDf['pair1Start'].astype('str') + '_' + outDf['pair1End'].astype('str')
    outDf['pair2'] += '_' + outDf['pair2Start'].astype('str') + '_' + outDf['pair2End'].astype('str')
    outDf['Pair'] = outDf['pair1'].astype('str') + '-' + outDf['pair2'].astype('str')
    outDf.drop(inplace=True, columns=['pair1Start', 'pair1End', 'pair2Start', 'pair2End', 'pair1', 'pair2'])
    
    #add all protein columns between 'chrom' and 'reads' (usually proteins start, end, window and distance)
    firstColumnToAdd = df.columns.get_loc('chrom') + 1
    endColumnToAdd = df.columns.get_loc('reads') #reads column is not added
    for i in range(firstColumnToAdd, endColumnToAdd):
        columnName = str(df.columns[i])
        outDf[columnName] = df.iloc[:,i]
        priorDf.loc[i,:] = [columnName, "Count"] 
    outDf['Count'] = df['reads']
    #multiply and rename distance column
    if 'distance' in outDf.columns:
        outDf['distance'] *= resolutionInt
        outDf.rename(columns={'distance': 'Distance'}, inplace=True)
        priorDf.loc[priorDf['name'] == 'distance', 'name'] = 'Distance'


    #write out the dataframes
    datasetFilename = os.path.join(outpath , cellLineStr + "_dataset.txt")
    priorFilename = os.path.join(outpath, cellLineStr + "_priors.txt")
    outDf.to_csv(datasetFilename, header=True, index=False, sep="\t")
    priorDf.to_csv(priorFilename, header=False, index=False, sep="\t")



if __name__=="__main__":
    convertForHicReg()