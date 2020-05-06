#!/usr/bin/env python3
import click
from itertools import product
from tqdm import tqdm
import json


_setAndProtein_options = [
    click.option('--baseFile','-bf', required=True,type=click.Path(writable=True),
        help='Base file where to store proteins and chromosomes for later use.'),
    click.option('--chromosomes', '-chs', default=None, show_default=True,help=\
                "If set, sets are only calculated for these chromosomes instead of all"),
]
_predict_base_options = [
    click.option('--predictionOutputDirectory', '-pod',default=None,\
                 type=click.Path(exists=True, writable=True),help='Output directory for'\
                 +' prediction files'),
    click.option('--resultsFilePath', '-rfp', default=None,show_default=True,\
              help='File where to store evaluation metrics. If not set'\
                +' no evaluation is executed'),
]
_predict_options = [
    click.option('--modelFilePath', '-mfp', required=True,type=click.Path(exists=True),\
              help='Choose model on which to predict'),
    click.option('--predictionSetPath','-psp', required=True,type=click.Path(exists=True),
        help='Data set that is to be predicted.'),
    click.option('--sigma', type=click.FloatRange(min=0.0,max=10.0), default=0.0),
    click.option('--noConvertBack', is_flag=True, help="Do not clamp predictions to normed input ranges")
]

_allpredict_options = [
    click.option('--modelDirectory', '-md', required=True,\
              help='Choose model directory'),
    click.option('--testSetDirectory','-tsd', required=True,type=click.Path(writable=True),
        help='Data set directory'),
    click.option('--chromosomes', '-chs', default=None, show_default=True,help=\
                "If set, sets are only calculated for these chromosomes instead of all"),
]
_protein_options = [
    click.option('--internalOutDir', '-iod', required=False, type=click.Path(writable=True, exists=True), \
                    help='path where internally used matrices will be stored'),
    click.option('--resolution' ,'-r', required=True, type=click.IntRange(min=1), \
                help = "Store resolution for analysis and documentation"),
    click.option('--cellType' ,'-ct', required=True, type=str,\
                help="Store cell type for analysis and documentation"),
    click.option('--matrixFile', '-mf',required=False,type=click.Path(exists=True),\
                 help='Input file with the whole HiC-matrix '),
    click.option('--correctMatrix', '-corr', type=bool, default=False, help="Correct the matrixFile using Knight-Ruiz method"),
    click.option('--chromsizeFile', '-csf', required=True, type=click.Path(exists=True), help="chrom.sizes file for binning the proteins")
]

_set_base_options = [
    click.option('--windowSize', '-ws', default=200, show_default=True,\
                help='Maximum distance between two potential interaction sites'),
    click.option('--datasetOutputDirectory', '-dod',required=True,type=click.Path(exists=True),\
                 help='Output directory for training set files'),
]
_allset_options = [
    click.option('--baseFile','-bf', required=True,type=click.Path(writable=True),
        help='Base file where to store proteins and chromosomes for later use.'),
    click.option('--setParamsFile', '-spf', required=True,\
              type=click.Path(exists=True)),
]
_set_options = [
    click.option('--mergeOperation','-mo',default='avg',\
                 type=click.Choice(['avg', 'max']),show_default=True,\
                 help='This parameter defines how the proteins are binned'),
    click.option('--divideProteinsByMean', type=bool, default=False, required=False, help="divide each protein signal by its mean"),
    click.option('--normalizeProteins', default=True, type=bool,\
                 show_default=True,\
                 help='Normalize protein signal values to the same range'),
    click.option('--normSignalValue', type=click.FloatRange(min=0.0), default=10.0, help="max. protein signal value after normalization"),
    click.option('--normSignalThreshold', type=click.FloatRange(min=0.0), default=0.1, help="after signal value normalization, set all values smaller than normSignalThreshold to 0."),
    click.option('--normalizeReadCounts', default=True, type=bool, help="Normalize HiC matrix read counts"),
    click.option('--normCountValue', type=click.FloatRange(min=0.0), default=10.0, help="max. read count value after normalization"),
    click.option('--normCountThreshold', type=click.FloatRange(min=0.0), default=0.0, help="after read count normalization, set all values smaller than normCountThreshold to 0."),
    click.option('--windowOperation', '-wo', default='avg',\
              type=click.Choice(['avg', 'max', 'sum']), show_default=True,\
                help='How should the proteins in between two base pairs be summed up'),
    click.option('--internalInDir', '-iid', type=click.Path(exists=True), \
                    help='path where internally used matrices are stored'),
    click.option('--smooth',required=False, type=click.FloatRange(min=0.0, max=10.0), default=0.0, help="standard deviation for gaussian smoothing of protein peaks; Zero means no smoothing"),                
    click.option('--method',required=False, type=click.Choice(['oneHot', 'multiColumn']), default='multiColumn', help="how to build the dataset. MultiColumn = 3 columns for each protein (start, window, end), OneHot = 3 columns (start, window, end) + one-hot encoding for the proteins"),
    click.option('--removeEmpty', required=False, type=bool, default=True, help="remove samples which have no protein data"),
    click.option('--noDiagonal', '-nd',required=False, type=click.IntRange(min=-1),default=-1,help="number of (side-)diagonals to ignore for training, default -1 (none), 0= main diagonal, 1= first side diagonal etc."),
    click.option('--printproteins', '-pp', required=False, type=bool, default=False, help="print protein plots"),
]
_train_options = [
    click.option('--trainDatasetFile', '-tdf',required=True,type=click.Path(exists=True, dir_okay=False, readable=True), help='dataset for training'),
    click.option('--noDist', required=False, type=bool, default=False, help="leave out distances when building the model, default False"),
    click.option('--noMiddle', required=False, type=bool, default=False, help="leave out middle proteins when building the model, default False"),
    click.option('--noStartEnd', required=False, type=bool, default=False, help="leave out start and end proteins when building the model, default False"),
    click.option('--weightBound1', '-wb1', required=False, type=click.FloatRange(min=0.0), default=0.0, help="samples within [weightBound1...weightBound2] will be emphasized; only relevant if ovsF > 0; default 0"),
    click.option('--weightBound2', '-wb2', required=False, type=click.FloatRange(min=0.0), default=0.0, help="samples within [weightBound1...weightBound2] will be emphasized; only relevant if ovsF > 0; default 0"),
    click.option('--ovsFactor', '-ovsF', required=False, type=click.FloatRange(min=0.0), default=0.0, help="factor by which the weights within the range between weightBound1/2 are multiplied, default=0 => no weighting"),
    click.option('--tadDomainFile', '-tads', required=False, type=click.Path(exists=True, dir_okay=False, readable=True), default=None, help="TAD domain file in bed format"),
    click.option('--weightingType', '-wt', type=click.Choice(choices=['reads', 'proteinFeatures', 'tads']), default='reads', help="compute weights based on reads (default) or protein feature values, only relevant if ovsF > 0"),
    click.option('--featList', '-fl', multiple=True, type=str, default=['0','12','24'], required=True, help="name of features according to which the weight is computed; default is 0, 12, 24; only relevant if wt=proteinFeatures and ovsF > 0"),    
    click.option('--plotTrees', required=False, type=bool, default=False, help="Plot decision trees, default False"),
    click.option('--splitTrainset', type=bool, default=False, help="Split Trainingset for a 5-fold Cross-Validation, i.e. return 5 models instead of 1; default False"),
    click.option('--useExtraTrees', type=bool, default=False, required=False, help="Use extra trees algorithm instead of random forests; default False")
]
_alltrain_options = [
    click.option('--setDirectory', '-sd',type=click.Path(exists=True),\
                 help='Input directory for training files', required=True,),
]
_train_base_options = [
    click.option('--modelOutputDirectory', '-mod',type=click.Path(exists=True, writable=True),\
                 help='Output directory for model files', required=True,),
    click.option('--conversion', '-co', default='none',\
              type=click.Choice(['standardLog', 'none']), show_default=True,\
                help='Define a conversion function for the read values'),
                ]
def protein_options(func):
    for option in reversed(_protein_options):
        func = option(func)
    for option in reversed(_setAndProtein_options):
        func = option(func)
    return func

def set_options(func):
    for option in reversed(_set_options):
        func = option(func)
    for option in reversed(_setAndProtein_options):
        func = option(func)
    for option in reversed(_set_base_options):
        func = option(func)
    return func

def predict_options(func):
    for option in reversed(_predict_base_options):
        func = option(func)
    for option in reversed(_predict_options):
        func = option(func)
    return func

def allpredict_options(func):
    for option in reversed(_predict_base_options):
        func = option(func)
    for option in reversed(_allpredict_options):
        func = option(func)
    return func

def train_options(func):
    for option in reversed(_train_options):
        func = option(func)
    for option in reversed(_train_base_options):
        func = option(func)
    return func

def alltrain_options(func):
    for option in reversed(_alltrain_options):
        func = option(func)
    for option in reversed(_train_base_options):
        func = option(func)
    return func

def allset_options(func):
    for option in reversed(_set_base_options):
        func = option(func)
    for option in reversed(_setAndProtein_options):
        func = option(func)
    for option in reversed(_allset_options):
        func = option(func)
    return func


def getBaseCombinations():
    params = {
        'mergeOperation': ["avg", "max"]
    }
    paramDict =  product(*params.values())
    for val in tqdm(list(paramDict), desc= 'Iterate parameter combinations' ):
        yield dict(zip(params, val))

def getCombinations(paramsfile):
    with open(paramsfile) as f:
       params = json.load(f)
    paramDict =  product(*params.values())
    for val in tqdm(list(paramDict), desc= 'Iterate parameter combinations' ):
        yield dict(zip(params, val))


