#!/usr/bin/env python3
import click
from itertools import product
from tqdm import tqdm
import json


class Mutex(click.Option):
    def __init__(self, *args, **kwargs):
        self.not_required_if:list = kwargs.pop("not_required_if")

        assert self.not_required_if, "'not_required_if' parameter required"
        kwargs["help"] = (kwargs.get("help", "") + "Option is mutually exclusive with " + ", ".join(self.not_required_if) + ".").strip()
        super(Mutex, self).__init__(*args, **kwargs)

    def handle_parse_result(self, ctx, opts, args):
        current_opt:bool = self.name in opts
        for mutex_opt in self.not_required_if:
            if mutex_opt in opts:
                if current_opt:
                    raise click.UsageError("Illegal usage: '" + str(self.name)\
                                           +"' is mutually exclusive with " + str(mutex_opt) + ".")
                else:
                    self.prompt = None
        return super(Mutex, self).handle_parse_result(ctx, opts, args)

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
    click.option('--baseFile','-bf', required=True,type=click.Path(exists=True),
        help='Base file used to create the predicted set'),
    click.option('--internalInDir', '-iid', required=False, type=click.Path(exists=True),
                help='path where internally used matrices are stored')

]
_predict_options = [
    click.option('--modelFilePath', '-mfp', required=True,type=click.Path(exists=True),\
              help='Choose model on which to predict'),
    click.option('--predictionSetPath','-psp', required=True,type=click.Path(exists=True),
        help='Data set that is to be predicted.'),
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
    click.option('--internalOutDir', '-iod', required=True, type=click.Path(writable=True), \
                    help='path where internally used matrices will be stored'),
    click.option('--resolution' ,'-r', required=True,\
                help = "Store resolution for analysis and documentation"),
    click.option('--cellType' ,'-ct', required=True, \
                help="Store cell type for analysis and documentation"),
    click.option('--matrixFile', '-mf',required=True,type=click.Path(exists=True),\
                 help='Input file with the whole HiC-matrix ')
]

_set_base_options = [
    click.option('--windowSize', '-ws', default=200, show_default=True,\
                help='Maximum distance between two basepairs'),
    click.option('--centromeresFile', '-cmf',show_default=True,
                 default=None,\
              type=click.Path(exists=True)),
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
    click.option('--peakColumn' ,'-pc', default=6,hidden=True),
    click.option('--mergeOperation','-mo',default='avg',\
                 type=click.Choice(['avg', 'max']),show_default=True,\
                 help='This parameter defines how the proteins are binned'),
    click.option('--normalize', default=False,\
                 show_default=True,\
                 help='Normalize protein signal values to a 0-1 range'),
    click.option('--ignoreCentromeres', type=bool, default=True,\
                 show_default=True,help='Cut out the centroid arms for training'),
    click.option('--windowOperation', '-wo', default='avg',\
              type=click.Choice(['avg', 'max', 'sum']), show_default=True,\
                help='How should the proteins in between two base pairs be summed up'),
    click.option('--internalInDir', '-iid', type=click.Path(exists=True), \
                    help='path where internally used matrices are stored')
]
_train_options = [
    click.option('--trainDatasetFile', '-tdf',\
                 required=True,\
                 help='File from which training is loaded'\
                 ,type=click.Path(exists=True)),
    click.option('--noDist', required=False, type=bool, default=False, help="leave out distances when building the model"),
    click.option('--noMiddle', required=False, type=bool, default=False, help="leave out middle proteins when building the model")
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
    click.option('--trees', type=click.IntRange(min=10, max=100, clamp=True), default=10, required=False),
    click.option('--maxFeat', type=click.Choice(['sqrt', 'none']), default='none', required=False)
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
        'mergeOperation': ["avg", "max"],
        'normalize': [True, False],
        'peakColumn': [4,6],
    }
    paramDict =  product(*params.values())
    for val in tqdm(list(paramDict), desc= 'Iterate parameter combinations' ):
        yield dict(zip(params, val))

def checkExtension(fileName, extension, option=None):
    if option:
        return ( fileName.endswith(extension) or fileName.endswith(option) )
    else:
        return fileName.endswith(extension)

def getCombinations(paramsfile):
    with open(paramsfile) as f:
       params = json.load(f)
    paramDict =  product(*params.values())
    for val in tqdm(list(paramDict), desc= 'Iterate parameter combinations' ):
        yield dict(zip(params, val))


