#!/usr/bin/env python3

from hicprediction.predict import executePrediction
from  hicprediction.configurations import *
import json

@allpredict_options
@click.command()
def predictAll(chromosomes, basefile, predictionoutputdirectory, resultsfilepath,
               modeldirectory, testsetdirectory):
    if chromosomes:
        chromosomeList = chromosomes.split(',')
    else:
        chromosomeList = range(1, 23)
    chromosomeList = [str(s) for s in chromosomeList]
    sets = dict()
    params = dict()
    for setPath in tqdm(os.listdir(testsetdirectory), desc="Load data sets once"):
        testSet, setParams = joblib.load( testsetdirectory +"/"+ setPath)
        sets[setPath] = testSet
        params[setPath] = setParams
    for modelpath in tqdm(os.listdir(modeldirectory), desc="Iterate models"):
        checkExtension(modeldirectory + "/" + modelpath, 'z')
        model, modelParams = joblib.load(modeldirectory + "/" +modelpath)
        # print(modelParams['chrom'][3:])
        # print(chromosomeList)
        if modelParams['chrom'][3:] in chromosomeList:
            model.set_params(verbose=1)
            for setPath in tqdm(os.listdir(testsetdirectory), desc="Iterate data sets"):
                # print(params[setPath]['chrom'])
                if params[setPath]['chrom'][3:] in chromosomeList:
                    executePrediction(model, modelParams,basefile,\
                    sets[setPath], params[setPath],predictionoutputdirectory, resultsfilepath)

if __name__ == '__main__':
    predictAll()
