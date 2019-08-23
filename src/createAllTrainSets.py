#!/usr/bin/env python3

from src.createTrainingSet import createTrainSet
from src.configurations import *

@allset_options
@click.command()
def createAllSets(basefile, centromeresfile, windowsize, datasetoutputdirectory):
    for setting in getSetCombinations():
        createTrainSet(None, datasetoutputdirectory,basefile,\
                    centromeresfile, setting['ignoreCentromeres'],
                    setting['normalize'], setting['equalize'],
                    setting['windowOperation'], setting['mergeOperation'],
                    windowsize, setting['peakColumn'])

def getSetCombinations():
    params = {
        'mergeOperation': ["avg", "max"],
        'windowOperation': ["avg"],
        'normalize': [True, False],
        'ignoreCentromeres': [True],
        'equalize': [False],
        'peakColumn': [6],
    }

    paramDict =  product(*params.values())
    for val in tqdm(list(paramDict), desc= 'Iterate parameter combinations' ):
        yield dict(zip(params, val))


if __name__ == '__main__':
    createAllSets()
