#!/usr/bin/env python3

from hicprediction.createTrainingSet import createTrainSet
from  hicprediction.configurations import *
import json

@allset_options
@click.command()
def createAllSets(basefile, centromeresfile, windowsize,
                  datasetoutputdirectory, setparamsfile):
    for setting in getSetCombinations(setparamsfile):
        createTrainSet(None, datasetoutputdirectory,basefile,\
                    centromeresfile, setting['ignoreCentromeres'],
                    setting['normalize'], setting['equalize'],
                    setting['windowOperation'], setting['mergeOperation'],
                    windowsize, setting['peakColumn'])

def getSetCombinations(setparamsfile):
    with open(setparamsfile) as f:
       params = json.load(f)

    paramDict =  product(*params.values())
    for val in tqdm(list(paramDict), desc= 'Iterate parameter combinations' ):
        yield dict(zip(params, val))

if __name__ == '__main__':
    createAllSets()
