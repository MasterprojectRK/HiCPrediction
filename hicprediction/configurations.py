#!/usr/bin/env python3
import click
from itertools import product
from tqdm import tqdm
import json


_setAndProtein_options = [
    click.option('--baseFile','-bf', required=True,type=click.Path(writable=True,dir_okay=False),
        help='Base file for binned ChIP-seq data on a per-chromosome basis for later use. Should be given .ph5 file extension'),
    click.option('--chromosomes', '-chs', default=None, show_default=True,help=\
                "If set, datasets are only calculated for these chromosomes instead of all."\
                    +"Specify chromosomes by numbers only, e.g. 'chs 1' or 'chs 1,3,17'"\
                    +"Only numerical chromosomes are allowed."\
                    +"Default: all numerical human chromsomes"),
]

_predict_base_options = [
    click.option('--predictionOutputDirectory', '-pod',default=None,\
                 type=click.Path(exists=True, writable=True, file_okay=False),\
                 help='Output directory for prediction files'),
    click.option('--resultsFilePath', '-rfp', default=None, type=click.Path(writable=False, dir_okay=False),\
              help='Filename where to store evaluation metrics. Optional. If not set, no evaluation is performed'),
]

_predict_options = [
    click.option('--modelFilePath', '-mfp', required=True,type=click.Path(exists=True, dir_okay=False, readable=True),\
              help="Hicprediction-model on which to predict. Models are created by calling 'training' and have a .z file extension."),
    click.option('--predictionSetPath','-psp', required=True,type=click.Path(exists=True, dir_okay=False),
        help="Dataset that is to be predicted. Datasets are created by calling 'createTrainingSet' and have a .z file extension"),
    click.option('--sigma', type=click.FloatRange(min=0.0,max=10.0), default=0.0, show_default=True,\
        help="Standard deviation for smoothing the predicted matrix with a 2D-Gaussian. Zero means no smoothing."),
    click.option('--noConvertBack', is_flag=True, show_default=True, help="Do not scale predictions to given input ranges")
]

_protein_options = [
    click.option('--internalOutDir', '-iod', required=False, type=click.Path(writable=True, exists=True, file_okay=False), \
                    help="Path where internally used matrices will be stored. Required if 'matrixFile' is given"),
    click.option('--resolution' ,'-r', required=True, type=click.IntRange(min=1), \
                help = "Resolution of matrices (e.g. '5000'), required for analysis and documentation"),
    click.option('--cellType' ,'-ct', required=True, type=str,\
                help="Cell type used for analysis and documentation, arbitrary string e.g. 'GM12878'"),
    click.option('--matrixFile', '-mf',required=False,type=click.Path(exists=True, readable=True, dir_okay=False),\
                 help='Hi-C matrix in cooler format; optional, required only for training, must then at least have all chromosomes specified by -chs option'),
    click.option('--correctMatrix', '-corr', type=bool, default=True, show_default=True, help="Correct the 'matrixFile' using Knight-Ruiz method"),
    click.option('--chromsizeFile', '-csf', required=True, type=click.Path(exists=True), help="chrom.sizes file is required for binning the proteins")
]

_set_base_options = [
    click.option('--windowSize', '-ws', default=200, show_default=True,\
                help='Maximum distance in bins between two potential interaction sites'),
    click.option('--datasetOutputDirectory', '-dod',required=True,type=click.Path(exists=True, writable=True, file_okay=False),\
                 help='Output directory for training set files'),
]

_set_options = [
    click.option('--mergeOperation','-mo',default='avg',\
                 type=click.Choice(['avg', 'max']),show_default=True,\
                 help='Proteins can be aggregated by taking the mean (avg) or the max signal value within each bin'),
    click.option('--divideProteinsByMean', type=bool, default=False, required=False, show_default=True, help="Divide each protein signal by its mean"),
    click.option('--normalizeProteins', type=bool, default=False, show_default=True, help='Scale protein signal values to value range [0...normSignalValue]'),
    click.option('--normSignalValue', type=click.FloatRange(min=0.0), default=1000.0, show_default=True, help="Max. protein signal value after scaling"),
    click.option('--normSignalThreshold', type=click.FloatRange(min=0.0), default=0.1, show_default=True, help="Set all values smaller than 'normSignalThreshold' to 0 after signal value scaling"),
    click.option('--normalizeReadCounts', type=bool, default=True, show_default=True, help="Scale Hi-C matrix read counts to value range [0...normCountValue]"),
    click.option('--normCountValue', type=click.FloatRange(min=0.0), default=1000.0, show_default=True, help="Max. read count value after scaling"),
    click.option('--normCountThreshold', type=click.FloatRange(min=0.0), default=0.0, show_default=True, help="Set all values smaller than 'normCountThreshold' to 0 after read count scaling"),
    click.option('--windowOperation', '-wo', default='avg',\
              type=click.Choice(['avg', 'max', 'sum']), show_default=True,\
                help='Window features can be computed by averaging, summing or taking the max across all bins within the window'),
    click.option('--internalInDir', '-iid', type=click.Path(exists=True, readable=True, file_okay=False), \
                    help="Path where internally used matrices are loaded from. These matrices are generated by 'createBaseFile' in cooler format and must not be renamed."),
    click.option('--smooth',required=False, type=click.FloatRange(min=0.0, max=10.0), default=0.0, show_default=True, help="Standard deviation for gaussian smoothing of protein peaks; Zero means no smoothing"),                
    click.option('--method',required=False, type=click.Choice(['oneHot', 'multiColumn']), default='multiColumn', show_default=True, help="How to build the dataset. MultiColumn = 3 columns for each protein (start, window, end) + distance, OneHot = 3 columns in total (start, window, end) + one-hot encoding for the proteins + distance"),
    click.option('--removeEmpty', required=False, type=bool, default=True, help="Invalidate samples which have no protein data"),
    click.option('--noDiagonal', '-nd',required=False, type=click.IntRange(min=-1),default=-1,help="Number of (side-)diagonals to ignore for training, default -1 (none), 0= main diagonal, 1= first side diagonal etc."),
    click.option('--printproteins', '-pp', required=False, type=bool, default=False, show_default=True, help="Print protein plots"),
]
_train_options = [
    click.option('--trainDatasetFile', '-tdf',required=True,type=click.Path(exists=True, dir_okay=False, readable=True), help="Hicprediction-dataset for training. Datasets are created by calling 'createTrainingSet' and have .z file extension"),
    click.option('--noDist', required=False, type=bool, default=False, show_default=True, help="Leave out distances when building the model"),
    click.option('--noMiddle', required=False, type=bool, default=False, show_default=True, help="Leave out window features when building the model"),
    click.option('--noStartEnd', required=False, type=bool, default=False, show_default=True, help="Leave out start- and end-features when building the model"),
    click.option('--weightBound1', '-wb1', required=False, type=click.FloatRange(min=0.0), default=0.0, show_default=True, help="Samples within [weightBound1...weightBound2] will be emphasized; only relevant if 'ovsF' > 0 and 'weightingType' != tads"),
    click.option('--weightBound2', '-wb2', required=False, type=click.FloatRange(min=0.0), default=0.0, show_default=True, help="Samples within [weightBound1...weightBound2] will be emphasized; only relevant if 'ovsF' > 0 and 'weightingType' != tads"),
    click.option('--ovsFactor', '-ovsF', required=False, type=click.FloatRange(min=0.0), default=0.0, show_default=True, help="Weight samples within weightBound1/2 or within TADs such that 'ovsF'=(weight sum of weighted samples)/(weight sum of unweighted samples), 0.0 = no weighting"),
    click.option('--tadDomainFile', '-tads', required=False, type=click.Path(exists=True, dir_okay=False, readable=True), default=None, help="TAD domain file in bed format for emphasizing samples within TADs; must be provided if 'weightingType' = tads"),
    click.option('--weightingType', '-wt', type=click.Choice(choices=['reads', 'proteinFeatures', 'tads']), default='reads', help="Compute weights for sample emphasizing based on reads, protein feature values or TADs, only relevant if 'ovsF' > 0"),
    click.option('--featList', '-fl', multiple=True, type=str, default=['0','12','24'], required=True, show_default=True, help="Name of features according to which the weights are computed; only relevant if 'weightingType' = proteinFeatures and 'ovsF' > 0"),    
    click.option('--plotTrees', required=False, type=bool, default=False, show_default=True, help="Plot decision trees (increases runtime)"),
    click.option('--splitTrainset', type=bool, default=False, show_default=True, help="Split Trainingset to do a 5-fold Cross-Validation, i.e. return 5 models instead of 1"),
    click.option('--useExtraTrees', type=bool, default=False, show_default=True, required=False, help="Use sklearn.ensemble extra trees algorithm instead of random forests")
]

_train_base_options = [
    click.option('--modelOutputDirectory', '-mod',type=click.Path(exists=True, writable=True, file_okay=False),\
                 help='Output directory for model files', required=True,),
    click.option('--conversion', '-co', default='none',\
              type=click.Choice(['standardLog', 'none']), show_default=True,\
                help='Conversion function for the interaction count values'),
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

def train_options(func):
    for option in reversed(_train_options):
        func = option(func)
    for option in reversed(_train_base_options):
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


