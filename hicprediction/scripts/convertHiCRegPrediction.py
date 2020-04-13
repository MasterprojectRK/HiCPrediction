import click
import pandas as pd
import numpy as np
import cooler
import os
import hicprediction.predict
from hicprediction.tagCreator import getResultFileColumnNames
import sklearn.metrics as metrics

@click.option('--resultsfile', '-r', type=click.Path(exists=True, readable=True), required=True, help="results file from HiC-Reg in text format")
@click.option('--resolution', '-res', type=click.IntRange(min=0), required=False, default=5000, help="resolution in bp")
@click.option('--outfolder', '-o', type=click.Path(exists=True, writable=True,file_okay=False, dir_okay=True),required=True, help="folder where conversion results (cooler, results csv file) will be placed")
@click.option('--traincelltype', '-tct', type=str, default="unknown", required=False, help="cell type for training")
@click.option('--predictioncelltype', '-pct', type=str, default="unknown", required=False, help="cell type for prediction")
@click.option('--trainchrom', '-tchr', type=str, default="unknown", required=False, help="training chromosome")
@click.command()
def convertHicReg(resultsfile, outfolder, resolution, traincelltype, predictioncelltype, trainchrom):
    try:
        resultsDf = pd.read_csv(resultsfile, delimiter="\t", index_col=False)
        resultsDf['pair1'] = [x.split("-")[0] for x in resultsDf['Column']]
        resultsDf['pair2'] = [x.split("-")[1] for x in resultsDf['Column']]
        resultsDf['first'] = [int(x.split("_")[1]) for x in resultsDf['pair1']]
        resultsDf['second'] = [int(x.split("_")[1]) for x in resultsDf['pair2']]
        resultsDf['chromosome'] = [x.split("_")[0] for x in resultsDf['pair1']]
        resultsDf.drop(columns=['Column', 'pair1', 'pair2'], inplace=True)
    except Exception as e:
        msg = str(e) + "\n"
        msg += "Could not read results file, maybe wrong format?"
        raise SystemExit(msg)
    print("successfully read HiC-reg input file")
    resultsDf['bin1_id'] = np.uint32(np.floor(resultsDf['first'] / resolution))
    resultsDf['bin2_id'] = np.uint32(np.floor(resultsDf['second'] / resolution))
    resultsDf['bin_distance'] = np.uint32((resultsDf['second'] - resultsDf['first'])/resolution)
    
    negMask = resultsDf['PredictedValue'] < 0
    if not resultsDf[negMask].empty:
        msg = "{:d} predictions were less than 0"
        msg = msg.format(resultsDf[negMask].shape[0])
        print(msg)
        resultsDf.loc[negMask, 'PredictedValue'] = 0.0

    predictionCsvFile = os.path.join(outfolder, "hicreg_results.csv")
    createCsvFromDf(pResultsDf=resultsDf, pResolution=resolution, 
                    pPredictionCellType=predictioncelltype, 
                    pTrainingCellType=traincelltype,
                    pTrainChrom=trainchrom, 
                    pTag=resultsfile, 
                    pOutfile=predictionCsvFile)
    print("created hicprediction csv results file {:s}".format(predictionCsvFile))

    predictionCoolerFile = os.path.join(outfolder, "hicreg_prediction.cool")
    targetCoolerFile = os.path.join(outfolder, "hicreg_target.cool")
    createCoolersFromDf(resultsDf, resolution, pPredictionOutfile=predictionCoolerFile, pTargetOutfile=targetCoolerFile )
    print("created cooler files {:s}, {:s}".format(predictionCoolerFile, targetCoolerFile))

def createCoolersFromDf(pResultsDf, pResolution, pPredictionOutfile, pTargetOutfile):
    #create the bins for cooler
    bins = pd.DataFrame(columns=['chrom','start','end'])
    maxPos = max(pResultsDf['bin1_id'].max(), pResultsDf['bin2_id'].max()) * pResolution + pResolution
    minPos = 0
    binStartList = list(range(minPos, maxPos, pResolution))
    binEndList = list(range(minPos + pResolution, maxPos, pResolution))
    binEndList.append(maxPos)
    bins['start'] = binStartList
    bins['end'] = binEndList
    bins['chrom'] = pResultsDf.loc[0, 'chromosome'] 
    #create the pixels / counts for predicted cooler
    pixels = pd.DataFrame(columns=['bin1_id','bin2_id','count'])
    pixels['bin1_id'] = pResultsDf['bin1_id']
    pixels['bin2_id'] = pResultsDf['bin2_id']
    pixels['count'] = pResultsDf['PredictedValue']
    pixels.sort_values(by=['bin1_id','bin2_id'],inplace=True)  
    #create the pixels / counts for target cooler
    targetPixels = pixels.copy(deep=True)
    targetPixels['count'] = pResultsDf['TrueValue']
    targetPixels.sort_values(by=['bin1_id','bin2_id'],inplace=True)
    #store the coolers
    cooler.create_cooler(pPredictionOutfile, bins=bins, pixels=pixels, dtypes={'count': np.float64})
    cooler.create_cooler(pTargetOutfile, bins=bins, pixels=targetPixels, dtypes={'count': np.float64})


def createCsvFromDf(pResultsDf, pResolution, pTag, pPredictionCellType, pTrainingCellType, pTrainChrom, pOutfile):
    columns = getResultFileColumnNames(sorted(list(pResultsDf.bin_distance.unique())))
    df = pd.DataFrame(columns=columns)
    df.set_index('Tag', inplace=True)

    indicesOPP, valuesOPP = hicprediction.predict.getCorrelation(pResultsDf, 'bin_distance', 'TrueValue', 'PredictedValue', 'pearson')

    ### calculate AUC
    aucScoreOPP = metrics.auc(indicesOPP, valuesOPP)
    corrScoreOP_P = pResultsDf[['TrueValue','PredictedValue']].corr(method= \
                'pearson').iloc[0::2,-1].values[0]
    corrScoreOP_S= pResultsDf[['TrueValue','PredictedValue']].corr(method= \
                'spearman').iloc[0::2,-1].values[0]
    #fill the dataframe
    df.loc[pTag, 'R2'] = metrics.r2_score(pResultsDf['TrueValue'], pResultsDf['PredictedValue'])
    df.loc[pTag, 'MSE'] = metrics.mean_squared_error(pResultsDf['TrueValue'], pResultsDf['PredictedValue'])
    df.loc[pTag, 'MAE'] = metrics.mean_absolute_error(pResultsDf['TrueValue'], pResultsDf['PredictedValue']),
    df.loc[pTag, 'MSLE'] = metrics.mean_squared_log_error(pResultsDf['TrueValue'], pResultsDf['PredictedValue']),
    df.loc[pTag, 'AUC_OP_P'] = aucScoreOPP
    df.loc[pTag, 'S_OP'] =  corrScoreOP_S 
    df.loc[pTag, 'P_OP'] = corrScoreOP_P 
    df.loc[pTag, 'Loss'] = 'MSE' 
    df.loc[pTag, 'resolution'] = pResolution
    df.loc[pTag, 'modelChromosome'] = pTrainChrom 
    df.loc[pTag, 'modelCellType'] = pTrainingCellType
    df.loc[pTag, 'predictionChromosome'] = pResultsDf.loc[0, 'chromosome'] 
    df.loc[pTag, 'predictionCellType'] = pPredictionCellType
    distStratifiedPearsonFirstIndex = df.columns.get_loc(0) 
    df.loc[pTag, distStratifiedPearsonFirstIndex:] = valuesOPP
    
    df = df.sort_values(by=['predictionCellType','predictionChromosome',
                            'modelCellType','modelChromosome', 'conversion',\
                            'Window','Merge', 'normalize'])
    df.to_csv(pOutfile)

if __name__=="__main__":
    convertHicReg()