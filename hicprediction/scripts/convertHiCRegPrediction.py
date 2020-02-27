import click
import pandas as pd
import numpy as np
import cooler
import os
import hicprediction.predict
import sklearn.metrics as metrics

@click.option('--resultsfile', '-r', type=click.Path(exists=True, readable=True), required=True, help="results file from HiC-Reg in text format")
@click.option('--resolution', '-res', type=int, required=False, default=5000, help="resolution in bp")
@click.option('--outfolder', '-o', type=click.Path(exists=True, writable=True,file_okay=False, dir_okay=True),required=True, help="folder where conversion results (cooler, results csv file) will be placed")
@click.command()
def convertHicReg(resultsfile, outfolder, resolution):
    try:
        resultsDf = pd.read_csv(resultsfile, delimiter="\t")
        resultsDf['first'] = [int(x.split("_")[1]) for x in resultsDf['Column']]
        resultsDf['second'] = [int(x.split("_")[4]) for x in resultsDf['Column']]
        resultsDf['chromosome'] = [x.split("_")[0] for x in resultsDf['Column']]
        resultsDf.drop(columns=['Column'], inplace=True)
    except Exception as e:
        msg = str(e) + "\n"
        msg += "Could not read results file, maybe wrong format?"
        raise SystemExit(msg)

    resultsDf['bin1_id'] = np.uint32(resultsDf['first'] / resolution)
    resultsDf['bin2_id'] = np.uint32(resultsDf['second'] / resolution)
    resultsDf['bin_distance'] = np.uint32(resultsDf['Distance'] / resolution)

    predictionCsvFile = os.path.join(outfolder, "hicreg_results.csv")
    createCsvFromDf(resultsDf, resolution, resultsfile, predictionCsvFile)



    predictionCoolerFile = os.path.join(outfolder, "hicreg_prediction.cool")
    targetCoolerFile = os.path.join(outfolder, "hicreg_target.cool")
    createCoolersFromDf(resultsDf, resolution, pPredictionOutfile=predictionCoolerFile, pTargetOutfile=targetCoolerFile )


def createCoolersFromDf(pResultsDf, pResolution, pPredictionOutfile, pTargetOutfile):
    #create the bins for cooler
    bins = pd.DataFrame(columns=['chrom','start','end'])
    maxPos = max(pResultsDf['first'].max(), pResultsDf['second'].max()) + pResolution
    minPos = min(pResultsDf['first'].min(), pResultsDf['second'].min())
    binStartList = list(range(minPos, maxPos, pResolution))
    binEndList = list(range(minPos + pResolution, maxPos, pResolution))
    binEndList.append(maxPos)
    bins['start'] = binStartList
    bins['end'] = binEndList
    bins['chrom'] = pResultsDf.loc[0, 'chromosome'] 
    print(bins)

    pixels = pd.DataFrame(columns=['bin1_id','bin2_id','count'])
    pixels['bin1_id'] = pResultsDf['bin1_id']
    pixels['bin2_id'] = pResultsDf['bin2_id']
    pixels['count'] = pResultsDf['PredictedValue']
    print(pixels)
    print("max1:", pixels['bin1_id'].max())
    print("max2:", pixels['bin2_id'].max())
    pixels.sort_values(by=['bin1_id','bin2_id'],inplace=True)  

    targetPixels = pixels.copy(deep=True)
    targetPixels['count'] = pResultsDf['TrueValue']
    targetPixels.sort_values(by=['bin1_id','bin2_id'],inplace=True)
    
    cooler.create_cooler(pPredictionOutfile, bins=bins, pixels=pixels, dtypes={'count': np.float64})
    cooler.create_cooler(pTargetOutfile, bins=bins, pixels=targetPixels, dtypes={'count': np.float64})

def createCsvFromDf(pResultsDf, pResolution, pTag, pOutfile):
    columns = ['Score', 
                    'R2',
                    'MSE', 
                    'MAE', 
                    'MSLE',
                    'AUC_OP_S',
                    'AUC_OP_P', 
                    'S_OP', 
                    'S_OA', 
                    'S_PA',
                    'P_OP',
                    'P_OA',
                    'P_PA',
                    'Window', 
                    'Merge',
                    'normalize',
                    'ignoreCentromeres',
                    'conversion', 
                    'Loss', 
                    'resolution',
                    'modelChromosome', 
                    'modelCellType',
                    'predictionChromosome', 
                    'predictionCellType']
    dists = sorted(list(pResultsDf.bin_distance.unique()))
    columns.extend(dists)
    columns.append(pTag)
    df = pd.DataFrame(columns=columns)
    df.set_index(pTag, inplace=True)

    indicesOPP, valuesOPP = hicprediction.predict.getCorrelation(pResultsDf, 'bin_distance', 'TrueValue', 'PredictedValue', 'pearson')

    ### calculate AUC
    aucScoreOPP = metrics.auc(indicesOPP, valuesOPP)
    corrScoreOP_P = pResultsDf[['TrueValue','PredictedValue']].corr(method= \
                'pearson').iloc[0::2,-1].values[0]
    corrScoreOA_P = 'nan'
    corrScoreOP_S= pResultsDf[['TrueValue','PredictedValue']].corr(method= \
                'spearman').iloc[0::2,-1].values[0]
    corrScoreOA_S= 'nan'
    #model parameters cell type, chromosome, window operation and merge operation may be lists
    #so generate appropriate strings for storage
    cols = [0, 
            metrics.r2_score(pResultsDf['TrueValue'], pResultsDf['PredictedValue']),
            metrics.mean_squared_error(pResultsDf['TrueValue'], pResultsDf['PredictedValue']),
            metrics.mean_absolute_error(pResultsDf['TrueValue'], pResultsDf['PredictedValue']),
            metrics.mean_squared_log_error(pResultsDf['TrueValue'], pResultsDf['PredictedValue']),
            0, 
            aucScoreOPP, 
            corrScoreOP_S, 
            corrScoreOA_S,
            0, 
            corrScoreOP_P, 
            corrScoreOA_P,
            0, 
            'unknown',
            'unknown',
            'unknown', 
            'unknown',
            'unknown', 
            'MSE', 
            pResolution,
            pResultsDf.loc[0, 'chromosome'], 
            'unknown target ct',
            pResultsDf.loc[0, 'chromosome'], 
            'unknown train ct']
    cols.extend(valuesOPP)
    df.loc[pTag] = cols
    df = df.sort_values(by=['predictionCellType','predictionChromosome',
                            'modelCellType','modelChromosome', 'conversion',\
                            'Window','Merge', 'normalize'])
    df.to_csv(pOutfile)

if __name__=="__main__":
    convertHicReg()